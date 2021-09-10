#!/usr/bin/env python3

import torch
import torchvision
import shutil


def my(*model_parts_terminated_by_device_name, **optional_parameters):
    """Summary or Description of the Function

    Parameters:
    model_parts_terminated_by_device_name (list of objects): One or multiple file names with model definition (OpenVINO IR, a framework specific file),
                            terminated by a device name ('CPU', 'GPU' etc.). Not serialized model representation from
                            a framework can be specified instead of file name.

    optional_parameters (dict of objects): Optional conversion parameters.

    Returns:
    int:Returning value

   """
    print(model_parts_terminated_by_device_name)
    #print(device)
    print(optional_parameters)

my(print, help, 'CPU', attr=7)

my('path', 'CPU', ATTR1=34, h='459803')
my('path', 3, device='CPU', ATTR1=34, h='459803')
my('path', device='CPU')
my('path', device='CPU')


dummy_input = torch.randn(10, 3, 224, 224)
alexnet = torchvision.models.alexnet(pretrained=True)
#print(type(model))
#exit(0)
#torch.onnx.export(model, dummy_input, "alexnet.onnx")

import sys

import openvino.inference_engine
import os.path
import os

def flatten(t):
    return [item for sublist in t for item in sublist]

def import_native_model (ie_core, *args, **kwargs):
    print('Trying to import native model')
    if len(args) == 0:
        raise ValueError('Missing model object when trying to import FW native model object') # something went wrong
    print(args[0], type(args[0]))

    # Probe different known FWs by searching in already imported modules
    if 'tensorflow' in sys.modules:
        # re-import
        # TODO: the same as reuse? efficient?
        import tensorflow
        # TODO: list all possible variants from TF here:
        tmp_model_file = '__temp_model_tensorflow.internal.openvino'
        if isinstance(args[0], tensorflow.python.keras.engine.sequential.Sequential):
            kwargs['saved_model_dir'] = tmp_model_file
            try:
                args[0].save(tmp_model_file)
                print('MODEL WAS SAVED')
                # reuse mo path
                return read_model(ie_core, **kwargs) # TODO: Don't loose args
            finally:
                print('TODO: REMOVE TEMPORARY FILE')
                shutil.rmtree(tmp_model_file)
    if 'torch' in sys.modules:
        import torch
        tmp_model_file = '__temp_model_tensorflow.internal.openvino.onnx'
        if issubclass(type(args[0]), torch.nn.Module):
            try:
                # We must receive input shapes in kwargs['input_shape']
                # TODO: expand to input and batch
                torch.onnx.export(args[0], (*kwargs['input_shape'], {}) if 'input_shape' in kwargs else None, tmp_model_file, opset_version=11)
                print('MODEL WAS EXPORTED AS AN ONNX MODEL')
                del kwargs['input_shape']
                return read_model(ie_core, tmp_model_file, **kwargs)
            finally:
                os.remove(tmp_model_file)

def import_mo (core, *args, **kwargs):
    path = args[0] if len(args) > 0 and isinstance(args[0], str) and os.path.exists(args[0]) else None
    nargs = kwargs
    if path is not None:
        nargs['input_model'] = path
    tmp_file_name = "__tmp_model_name.openvino.internals"
    nargs['model_name'] = tmp_file_name
    nargs['silent'] = None
    if 'input_model' in nargs or 'saved_model_dir' in nargs:
        try:
            # TODO: Implement in a smart way considering the fact that MO emits intermediate IR as one of the backend steps
            from mo.subprocess_main import subprocess_main  # pylint: disable=no-name-in-module
            subprocess_main(framework=None, new_args=flatten([['--' + k, str(v)] if v is not None else ['--' + k] for k, v in nargs.items()]))
            network = core.read_network(tmp_file_name + '.xml')
            return network
        finally:
            os.remove(tmp_file_name + '.xml')
            os.remove(tmp_file_name + '.bin')
            os.remove(tmp_file_name + '.mapping')

def import_runtime (core, *args, **kwargs):
    if len(args) > 0 and isinstance(args[0], str):
        if not args[0].endswith('.pb'):  # TODO: Extend it by more rules based on file name
            try:
                # Try to use RT load first
                network = core.read_network(*args)
                return network
            except Exception as error:
                print('FAILED TO LOAD WITH Core.read_network. Trying to use MO')
                print(error)
                return import_mo(core, *args, **kwargs)
        else:
            return import_mo(core, *args, **kwargs)
    else:
        input_model_keys = ['input_model', 'saved_model_dir', 'input_proto', 'input_checkpoint', 'input_symbol', 'input_meta_graph']
        if len(set(input_model_keys).intersection(set(kwargs.keys()))) > 0:
            return import_mo(core, *args, **kwargs)
        else:
            return import_native_model(core, *args, **kwargs)


def read_model (ie_core, *args, **kwargs):
    if len(args) > 0:
        # There are can be two cases:
        #   (1) legacy usage with up to 2 arguments and without additional key-value arguments
        #   (2) with additional key-value parameters

        # TODO: Implement smarter logic for routing to MO, RT FE or FW export

        # If there is at least one key from this list, then model_optimizer is required for conversion
        model_optimizer_only_keys = ['saved_model_dir', 'inputs', 'outputs', 'input_shape']  # etc.

        return import_runtime(core, *args, **kwargs)
    else:
        # Backward compatibility to the old read_network arg names
        # TODO: Remove if decide the new API should have it
        if 'model' in kwargs.keys():
            assert 'input_model' not in kwargs.keys()
            kwargs['input_model'] = kwargs['model']
            del kwargs['model']

        if 'weights' in kwargs.keys():
            assert 'input_weights' not in kwargs.keys()
            kwargs['input_weights'] = kwargs['weights']
            del kwargs['weights']

        args = []

        if 'input_model' in kwargs.keys():
            args.append(kwargs['input_model'])

        if 'input_weights' in kwargs.keys():
            args.append(kwargs['input_weights'])

        return import_runtime(core, *args, **kwargs)


class MyCore(openvino.inference_engine.IECore):
    def read_model (self, *args, **kwargs):
        return read_model(self, *args, **kwargs)

core = openvino.inference_engine.IECore()

#################################
# Imagine that core = openvino.inference_engine.IECore()

model = core.read_network("/localdisk/slyalin/openvino_github/openvino_7/model-optimizer/resnet_v2_50.xml")
model.serialize("test.xml", "test.bin")
print(model)

model = core.read_network(input_model="/localdisk/slyalin/openvino_github/openvino_7/model-optimizer/resnet_v2_50.xml")
model.serialize("test.xml", "test.bin")
print(model)


#model = core.read_model("/localdisk/slyalin/models-temp/resnet_v2_50.pb", b=1)
#model.serialize("test.xml", "test.bin")
#print(model)

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions
tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

print(probability_model(x_test[:5]))

#core = openvino.inference_engine.IECore()
network = core.read_network(probability_model, batch=5)
executable = core.load_network(network, 'CPU')

print(executable.infer({'sequential_input': x_test[:5]}))

network_torch = core.read_network(alexnet, input_shape=[dummy_input]) # TODO: Replace dummy_input by just a shape
network_torch.serialize('exported_from_pytorch.xml')
