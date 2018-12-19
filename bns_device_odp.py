#!/usr/bin/python
# -*- coding: utf-8
#
# Copyright 2016 Julio Mateos Langerak (julio.mateos-langerak@igh.cnrs.fr)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Meadowlark-Boulder Non Linear Spacial Light Modulator device interface.

This module provides a wrapper for Meadowlark's SDK interface that allows
a ODP SLM and all its settings to be exposed to python.

The interface has been implemented using cffi API 'in line'
As this is a timing critical device it might be important to implement it using ABI mode.
"""

from cffi import FFI
# from imageio import imread
from scipy.misc import imread
import numpy as np
import threading
import os

# TODO: We should ge this from the config or using the os tools
# We import the headers definitions used by cffi from a file in order to avoid copyright issues
# Define a buffer of 1024 to hold the path of the LUTs
PATHS_BUFFER_SIZE = 1024
HEADER_DEFINITIONS = "C:\\Users\\omxt\\PycharmProjects\\bnsdevice\\Blink_SDK_C_wrapper_defs.h"
BLINK_SDK_DLL_PATH = "Blink_SDK_C.dll"
LUTS_PATH = b"C:\\Users\\omxt\\PycharmProjects\\bnsdevice\\LUT_files"
LUTS = {473: b'slm4039_at473_regional_encrypt.txt',
        532: b'slm4039_at532_regional_encrypt.txt',
        635: b'slm4039_at635_regional_encrypt.txt'}
DEFAULT_LUT_FILE = b"SLM_lut.txt"
DEFAULT_OVERDRIVE_LUT_FILE = b'slm4039_at473_regional_encrypt.txt'
PHASE_CALIBRATION_FILES = b"C:\\Users\\omxt\\PycharmProjects\\bnsdevice\\Phase_calibration_files"
BIT_DEPTH = 8
SLM_RESOLUTION = 512
USE_ODP = True
IS_NEMATIC_TYPE = True
RAM_WRITE_ENABLE = True
USE_GPU = True
TRIGGER_TIMEOUT_MS = 0
MAX_TRANSIENTS = 20
# Overdrive  SLMs  the  true  frames  parameter  should  be  set  to  5.
# For non-overdrive  operation  true frames should be set to 3
TRUE_FRAMES = 5


CLASS_NAME = "BNSDevice_ODP"


class BNSDevice_ODP(threading.Thread):
    """
    This class represents the BNS device
    Important note: The header defs is a text file containing the headers from
    the Blink_SDK.h file with some modifications. Namely all #include and #ifdef have been
    removed.

    :param header_definitions: Absolute path to the header definitions from the SDK.
    :param blink_sdk_dll_path: name of, or absolute path to, the SID4_SDK.dll file
    :param luts_path: Absolute path to the LUT files.
    :param default_lut_file:
    :param phase_calibration_files:
    :param bit_depth:
    :param slm_resolution:
    :param is_nematic_type:
    :param RAM_write_enable:
    :param use_GPU:
    :param max_transients:
    :param true_frames:
    """
    def __init__(self,
                 header_definitions=HEADER_DEFINITIONS,
                 blink_sdk_dll_path=BLINK_SDK_DLL_PATH,
                 luts_path=LUTS_PATH,
                 luts=LUTS,
                 default_lut_file=DEFAULT_LUT_FILE,
                 default_overdrive_lut_file=DEFAULT_OVERDRIVE_LUT_FILE,
                 phase_calibration_files=PHASE_CALIBRATION_FILES,
                 bit_depth=BIT_DEPTH,
                 slm_resolution=SLM_RESOLUTION,
                 use_odp = USE_ODP,
                 is_nematic_type=IS_NEMATIC_TYPE,
                 RAM_write_enable=RAM_WRITE_ENABLE,
                 use_GPU=USE_GPU,
                 trigger_timeout_ms=TRIGGER_TIMEOUT_MS,
                 max_transients=MAX_TRANSIENTS,
                 true_frames=TRUE_FRAMES):

        # Container for the handler
        self.slm_handle = None

        # COntainer to get the return from function calls
        self._r = None

        # Path for the LUT files
        self.luts_path = luts_path
        self.luts = luts
        self.default_overdrive_lut_file = default_overdrive_lut_file
        self.default_lut_file = default_lut_file

        # Get the SDK library
        self.header_definitions = header_definitions

        try:
            with open(self.header_definitions, 'r') as self.header_definitions:
                self.cdef_from_file = self.header_definitions.read()
        except FileNotFoundError:
            raise Exception('Unable to find "%s" header file.' % self.header_definitions)
        except IOError:
            raise Exception('Unable to open "%s"' % self.header_definitions)
        # finally:
        #     if self.cdef_from_file == '' or None:
        #         print('File "%s" is empty' % self.header_definitions)
        #         exit(3)

        # Create here the interface to the SDK
        self.ffi = FFI()
        self.ffi.cdef(self.cdef_from_file, override=True)
        self.blink_sdk = self.ffi.dlopen(blink_sdk_dll_path)

        # OverDrive Plus parameters
        self.use_odp = use_odp
        self.use_GPU = self.ffi.cast("int", use_GPU)
        self.max_transients = self.ffi.cast("int", max_transients)
        self.transient_images = None

        # Basic SLM parameters
        self.true_frames = self.ffi.cast('int', true_frames)
        if use_odp:
            self.default_static_regional_lut_file = self.ffi.new('char[]', os.path.join(luts_path, self.default_overdrive_lut_file))
        else:
            self.default_static_regional_lut_file = self.ffi.new('char[]', os.path.join(luts_path, self.default_lut_file))
        self.bit_depth = self.ffi.cast("unsigned int", bit_depth)
        self.slm_resolution = self.ffi.cast("unsigned int", slm_resolution)
        self.is_nematic_type = self.ffi.cast("int", is_nematic_type)
        self.RAM_write_enable = self.ffi.cast("int", RAM_write_enable)

        # Blank calibration image
        self.cal_image = np.full((slm_resolution, slm_resolution), (2**bit_depth)-1, 'uint' + str(bit_depth))

        # Boolean showing initialization status.
        self.haveSLM = False  # TODO: see if we can replace by constructed_okay
        self.power_state = self.ffi.cast("int", 0)

        # Boolean to control a running sequence and a holder for a thread
        self._sequence_running = False
        self._sequence_index = 1
        self.t = None

        # Boolean to control triggers use
        self.wait_for_trigger = self.ffi.cast("int", 0)
        self.trigger_timeout_ms = self.ffi.cast("unsigned int", trigger_timeout_ms)
        self.external_pulse = self.ffi.cast("int", 1)

        # Container to get the output of SDK function calls and errors
        #self._r = self.ffi.cast('int', 0)
        self.error = self.ffi.new('char[]', PATHS_BUFFER_SIZE)

        # Data type to store images.
        self.image_size = self.ffi.cast('unsigned int', int(self.bit_depth) * (int(self.slm_resolution) * int(self.slm_resolution)))

        self.num_boards_found = self.ffi.new("unsigned int *", 0)
        self.constructed_okay = self.ffi.new("int *", True)
        self.board = self.ffi.cast("int", 1)

        # Some variables used down the road
        self.phase_calibration_files = phase_calibration_files

    # DECORATORS
    # decorator definition for methods that require an SLM
    def requires_slm(func):
        def wrapper(self, *args, **kwargs):
            if not self.constructed_okay:
                raise Exception("SLM is not initialized.")
            else:
                return func(self, *args, **kwargs)

        return wrapper

    # PROPERTIES
    @property
    @requires_slm
    def curr_seq_image(self):
        return self._sequence_index

    @property
    @requires_slm
    def power(self):
        return self.power_state

    @power.setter
    @requires_slm
    def power(self, value):
        temp_power = self.power_state
        self.power_state = value
        try:
            self.blink_sdk.SLM_power(self.slm_handle, self.power_state)
        except:
            self.power_state = temp_power
            raise Exception(self.get_last_error())

    @property
    @requires_slm
    def temperature(self):  # TODO: this is not implemented
        return 20

    # METHODS
    ## Don't call this unless an SLM was initialised:  if you do, the next call
    # can open a dialog box from some other library down the chain.
    @requires_slm
    def cleanup(self):
        try:
            self.blink_sdk.Delete_SDK(self.slm_handle)
        except:
            pass
        self.constructed_okay[0] = 0
        self.haveSLM = False

    def initialize(self):
        ## Need to unload and reload the DLL here.
        # Otherwise, the DLL can open an error window about having already
        # initialized another DLL, which we won't see on a remote machine.

        # Initialize the library, looking for nematic SLMs.
        try:
            self.slm_handle = self.blink_sdk.Create_SDK(self.bit_depth,
                                                        self.num_boards_found,
                                                        self.constructed_okay,
                                                        self.is_nematic_type,
                                                        self.RAM_write_enable,
                                                        self.use_GPU,
                                                        self.max_transients,
                                                        self.default_static_regional_lut_file)
            if self.num_boards_found[0] == 0:
                raise Exception("No SLM device found.")
            elif self.num_boards_found[0] > 1:
                raise Exception("More than one SLM device found. This module can only handle one device.")
            else:
                if self.constructed_okay[0] == -1:
                    raise Exception("SLM constructor did not succeed.")
                else:
                    self.haveSLM = True
        except:
            raise Exception('Could not Initialize')

        # Turn off external trigger
        self.wait_for_trigger = 0

        # This is required after initialization
        self.set_true_frames(self.true_frames)

        # Load the default LUT
        if self.use_odp:
            self.load_lut(os.path.join(self.luts_path, self.default_overdrive_lut_file))
        else:
            self.load_lut(os.path.join(self.luts_path, self.default_lut_file))

        # Load a white image
        self.write_image(self.cal_image)

    @requires_slm
    def load_lut(self, filename):
        # We assume that the SLM has been initialized and that the linear LUT is already loaded on HW
        if type(filename) != bytes:
            filename = filename.encode()
        lut_file = self.ffi.new('char[]', filename)
        if self.use_odp:
            """When using ODP the regional calibration is used to linearize the regional response
            of the LC to voltage. The global calibration, which is applied in hardware should be 
            disabled by loading a linear LUT to the hardware."""
            self._r = self.blink_sdk.Load_linear_LUT(self.slm_handle,
                                                     self.board)
            if int(self._r):
                raise Exception(self.get_last_error())

            self._r = self.blink_sdk.Load_overdrive_LUT_file(self.slm_handle,
                                                             lut_file)
            if int(self._r):
                raise Exception(self.get_last_error())

        else:
            self._r = self.blink_sdk.Load_LUT_file(self.slm_handle,
                                                   self.board,
                                                   lut_file)
            if int(self._r):
                raise Exception(self.get_last_error())

        if self.use_odp:
            self.blink_sdk.Load_overdrive_LUT_file(self.slm_handle,
                                                   lut_file)
        else:
            self.blink_sdk.Load_LUT_file(self.slm_handle,
                                         self.board,
                                         lut_file)

    def load_wavelength_lut(self, wavelength):
        """Loads the LUT to the SLM that fits the best for a specified wavelength"""
        # Load the default LUT
        lut_wavelengths = self.luts.keys()
        nearest = min(lut_wavelengths, key=lambda x: abs(x - wavelength))
        lut_file = os.path.join(self.luts_path, self.luts[nearest])
        self.load_lut(lut_file)
        return None

    def write_cal(self, type, calImage):
        """A pass through for old SDK compatibility"""
        return self.write_image(calImage)

    @requires_slm
    def write_image(self, image, external_trigger=False):
        # This function loads an image to the SLM
        if self._sequence_running:
            raise Exception('Sequence is running. Cannot write single image')

        self.wait_for_trigger = external_trigger

        image = self.transform_16_to_8_bit(image)

        if self.use_odp:
            self._r = self.blink_sdk.Write_overdrive_image(self.slm_handle,
                                                           self.board,
                                                           self.ffi.from_buffer(image),
                                                           self.wait_for_trigger,
                                                           self.external_pulse,
                                                           self.trigger_timeout_ms)
            if int(self._r):
                raise Exception(self.get_last_error())

        else:
            self._r = self.blink_sdk.Write_image(self.slm_handle,
                                                 self.board,
                                                 self.ffi.from_buffer(image),
                                                 self.image_size,
                                                 self.wait_for_trigger,
                                                 self.external_pulse,
                                                 self.trigger_timeout_ms)
            if int(self._r):
                raise Exception(self.get_last_error())

    @requires_slm
    def compute_transients(self, image):
        print('Computing Transients')
        byte_count = self.ffi.new('unsigned int*', 0)
        self.blink_sdk.Calculate_transient_frames(self.slm_handle,
                                                  self.ffi.from_buffer(image),
                                                  byte_count)
        transients = self.ffi.new('unsigned char[]', byte_count[0])
        self.blink_sdk.Retrieve_transient_frames(self.slm_handle,
                                                 transients)
        return transients

    @requires_slm
    def load_sequence(self, image_wavelength_list):
        if len(image_wavelength_list) < 2:
            raise Exception("load_sequence expects a list of two or more " \
                            "images - it was passed %s images." % len(image_wavelength_list))
        # We pre-compute here the transient images
        # Verify that the calculation engine is properly loaded
        if self.blink_sdk.Is_slm_transient_constructed(self.slm_handle) < 0:
            raise Exception('SLM transient calculation engine not properly constructed')
        # Empty the list of transient images
        self.transient_images = []
        current_wavelength = None
        if self.use_odp:
            for image, wavelength in image_wavelength_list:
                if wavelength != current_wavelength:
                    self.load_wavelength_lut(wavelength)
                    current_wavelength = wavelength
                if type(image) is np.ndarray:
                    image = self.transform_16_to_8_bit(image)
                    self.transient_images.append(self.compute_transients(image))
                else:
                    raise Exception('Sequence of images is not in teh right format')
        else:
            for image in image_wavelength_list:  # In this case it is only an images list
                if type(image) is np.ndarray:
                    image = self.transform_16_to_8_bit(image)
                    self.transient_images.append(self.compute_transients(image))
                else:
                    raise Exception('Sequence of images is not in teh right format')

    @requires_slm
    def start_sequence(self, external_trigger=True):
        print('Starting sequence')
        self.wait_for_trigger = external_trigger
        self._sequence_running = True
        self.t = threading.Thread(target=self._run_sequence)
        self.t.start()

    def _run_sequence(self):
        while self._sequence_running:
            if self.transient_images:
                self._sequence_index = 1
                for transients in self.transient_images:
                    if self._sequence_running:
                        self._r = self.blink_sdk.Write_transient_frames(self.slm_handle,
                                                                        self.board,
                                                                        transients,
                                                                        self.wait_for_trigger,
                                                                        self.external_pulse,
                                                                        self.trigger_timeout_ms)
                        # print(self._sequence_index)
                        self._sequence_index += 1
                        if int(self._r):
                            print(self.get_last_error())
                            self._sequence_running = False
                            return
                    else:
                        return

    @requires_slm
    def stop_sequence(self):
        self._sequence_running = False
        self.blink_sdk.Stop_sequence(self.slm_handle)
        self.t.join()

    def transform_16_to_8_bit(self, array):
        coef = np.array([np.iinfo('uint8').max / np.iinfo('uint16').max])
        return np.multiply(array, coef).astype('uint8')

    def read_tiff(self, file_path):
        if type(file_path) != bytes:
            file_path = file_path.encode()

        return imread(file_path)

    def set_external_trigger_timeout(self, timeout_ms):
        self.trigger_timeout_ms = timeout_ms

    @requires_slm
    def set_sequencing_framrate(self, frame_rate):
        # Not implemented in Blink_sdk
        pass

    @requires_slm
    def set_true_frames(self, true_frames):
        self.true_frames = true_frames
        self.blink_sdk.Set_true_frames(self.slm_handle,
                                       self.true_frames)

    @requires_slm
    def get_last_error(self):
        return self.ffi.string(self.blink_sdk.Get_last_error_message(self.slm_handle))
