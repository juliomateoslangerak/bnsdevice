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
from imageio import imread
import numpy as np
import os

# TODO: We should ge tthis from the config or using the os tools
# We import the headers definitions used by cffi from a file in order to avoid copyright issues
# Define a buffer of 1024 to hold the path of the LUTs
PATHS_BUFFER_SIZE = 1024
HEADER_DEFINITIONS = "C:\\Users\\omxt\\PycharmProjects\\bnsdevice\\Blink_SDK_C_wrapper_defs.h"
BLINK_SDK_DLL_PATH = "Blink_SDK_C.dll"
LUTS_PATH = b"C:\\Users\\omxt\\PycharmProjects\\bnsdevice\\LUT_files"
DEFAULT_LUT_FILE = b"SLM_lut.txt"
DEFAULT_OVERDRIVE_LUT_FILE = b'slm4039_at473_regional_encrypt.txt'
PHASE_CALIBRATION_FILES = b"C:\\Users\\omxt\\PycharmProjects\\bnsdevice\\Phase_calibration_files"
BIT_DEPTH = 8
SLM_RESOLUTION = 512
USE_ODP = True
IS_NEMATIC_TYPE = True
RAM_WRITE_ENABLE = True
USE_GPU = True
TRIGGER_TIMEOUT_MS = 5000
MAX_TRANSIENTS = 20
# Overdrive  SLMs  the  true  frames  parameter  should  be  set  to  5.
# For non-overdrive  operation  true frames should be set to 3
TRUE_FRAMES = 5

CLASS_NAME = "BNSDevice"  # TODO: See if we change the name


class BNSDevice:
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
        self.slm = None

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
            self.default_static_regional_lut_file = self.ffi.new('char[]', os.path.join(luts_path, default_overdrive_lut_file))
        else:
            self.default_static_regional_lut_file = self.ffi.new('char[]', os.path.join(luts_path, default_lut_file))
        self.bit_depth = self.ffi.cast("unsigned int", bit_depth)
        self.slm_resolution = self.ffi.cast("unsigned int", slm_resolution)
        self.is_nematic_type = self.ffi.cast("int", is_nematic_type)
        self.RAM_write_enable = self.ffi.cast("int", RAM_write_enable)

        # Blank calibration image
        self.cal_image = np.full((slm_resolution, slm_resolution), (2**bit_depth)-1, 'uint' + str(bit_depth))

        # Boolean showing initialization status.
        self.haveSLM = False  # TODO: see if we can replace by constructed_okay

        # Boolean to control triggers use
        self.wait_for_trigger = self.ffi.cast("int", 0)
        self.trigger_timeout_ms = self.ffi.cast("unsigned int", trigger_timeout_ms)
        self.external_pulse = self.ffi.cast("int", 1)

        # Container to get the errors
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
            if not self.haveSLM:
                raise Exception("SLM is not initialized.")
            else:
                return func(self, *args, **kwargs)

        return wrapper

    def check_for_error(func):
        def get_error(self, *args, **kwargs):
            error = func(self, *args, **kwargs)
            if error:
                raise Exception(f'Error calling {func.__name__}: {self.get_last_error()}')

        return get_error(self, *args, **kwargs)


    # PROPERTIES
    @property
    @requires_slm
    def curr_seq_image(self):
        return self.lib.GetCurSeqImage(c_int(0))

    @property
    @requires_slm
    def power(self):
        return self.lib.GetSLMPower(c_int(0))

    @power.setter
    @requires_slm
    def power(self, value):
        self.blink_sdk.SLM_power(self.handle, value)

    @property
    @requires_slm
    def temperature(self):
        return self.lib.GetInternalTemp(c_int(0))

    # METHODS
    ## Don't call this unless an SLM was initialised:  if you do, the next call
    # can open a dialog box from some other library down the chain.
    @requires_slm
    def cleanup(self):
        try:
            self.blink_sdk.Delete_SDK(self.slm)
        except:
            pass
        self.haveSLM = False

    def initialize(self):
        ## Need to unload and reload the DLL here.
        # Otherwise, the DLL can open an error window about having already
        # initialized another DLL, which we won't see on a remote machine.

        # Initialize the library, looking for nematic SLMs.
        try:
            self.slm = self.blink_sdk.Create_SDK(self.bit_depth,
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
            """When using ODP the regional calibration is used to linearize the regional response
            of the LC to voltage. The global calibration, which is applied in hardware should be 
            disabled by loading a linear LUT to the hardware."""
            self.blink_sdk.Load_linear_LUT(self.slm,
                                           self.board)
            self.blink_sdk.Load_overdrive_LUT_file(self.slm,
                                                   self.default_static_regional_lut_file)
        else:
            self.blink_sdk.Load_LUT_file(self.slm,
                                         self.board,
                                         self.default_static_regional_lut_file)

        # Load a white image
        if self.use_odp:
            self.blink_sdk.Write_overdrive_image(self.slm,
                                                 self.board,
                                                 self.ffi.from_buffer(self.cal_image),
                                                 self.wait_for_trigger,
                                                 self.external_pulse,
                                                 self.trigger_timeout_ms)
        else:
            self.blink_sdk.Write_image(self.slm,
                                       self.board,
                                       self.ffi.from_buffer(self.cal_image),
                                       self.image_size,
                                       self.wait_for_trigger,
                                       self.external_pulse,
                                       self.trigger_timeout_ms)

    @requires_slm
    def load_lut(self, filename):
        # We assume that the SLM has been initialized and that the linear LUT is already loaded on HW
        lut_file = self.ffi.new('char[]', os.path.join(luts_path, filename.encode()))
        if self.use_odp:
            self.blink_sdk.Load_overdrive_LUT_file(self.slm,
                                                   lut_file)
        else:
            self.blink_sdk.Load_LUT_file(self.slm,
                                         self.board,
                                         lut_file)

    @requires_slm
    def compute_transients(self, image):
        byte_count = self.ffi.new('unsigned int*', 0)
        self.blink_sdk.Calculate_transient_frames(self.slm,
                                                  self.ffi.from_buffer(image),
                                                  byte_count)
        transients = self.ffi.new('unsigned char[]', byte_count)
        self.blink_sdk.Retrieve_transient_frames(self.slm,
                                                 transients)
        return transients

    @requires_slm
    def load_sequence(self, image_list):
        if len(image_list) < 2:
            raise Exception("load_sequence expects a list of two or more " \
                            "images - it was passed %s images." % len(image_list))
        # We pre-compute here the transient images
        if self.blink_sdk.Is_slm_transient_constructed(self.slm) < 0:
            raise Exception('SLM transient calculation engine not properly constructed')
        # Empty the list of transient images
        self.transient_images = []
        for image in image_list:
            if type(image) is np.ndarray:
                self.transient_images.append(self.compute_transients(image))
            else:
                raise Exception('Sequence of images is not in teh right format')

    @requires_slm
    def start_sequence(self, external_trigger=True):
        self.wait_for_trigger = external_trigger
        for transients in self.transient_images:
            self.blink_sdk.Write_transient_frames(self.slm,
                                                  self.board,
                                                  transients,
                                                  self.wait_for_trigger,
                                                  self.external_pulse,
                                                  self.trigger_timeout_ms)                                                 )

    def read_tiff(self, file_path):

        return imread(file_path.encode())

    @requires_slm
    def set_sequencing_framrate(self, frame_rate):
        # Not implemented in Blink_sdk
        pass

    @requires_slm
    def set_true_frames(self, true_frames):
        self.true_frames = true_frames
        self.blink_sdk.Set_true_frames(self.slm,
                                       self.true_frames)


    @requires_slm
    def get_last_error(self):
        return self.ffi.string(self.blink_sdk.Get_last_error_message(self.slm))

############# Non ODP code####################
# import os, sys
#
#
# bnsdatatype = ctypes.c_uint16
#
# #
# class BNSDevice(object):
#     """ Enables calls to functions in BNS's Interface.dll.
#
#     === BNS Interface.dll functions ===
#     + indicates equivalent python implementation here
#     o indicates calls from non-equivalent python code here
#
#     ==== Documented ====
#     + int Constructor (int LCType={0:FLC;1:Nematic})
#     + void Deconstructor ()
#     + void ReadTIFF (const char* FilePath, unsigned short* ImageData,
#                      unsigned int ScaleWidth, unsigned int ScaleHeight)
#     + void WriteImage (int Board, unsigned short* Image)
#     + void LoadLUTFile (int Board, char* LUTFileName)
#     o void LoadSequence (int Board, unsigned short* Image, int NumberOfImages)
#     + void SetSequencingRate (double FrameRate)
#     + void StartSequence ()
#     + void StopSequence ()
#     + bool GetSLMPower (int Board)
#     + void SLMPower (int Board, bool PowerOn)
#     + void WriteCal (int Board, CAL_TYPE Caltype={WFC;NUC},
#                      unsigned char* Image)
#       int ComputeTF (float FrameRate)
#     + void SetTrueFrames (int Board, int TrueFrames)
#
#     ==== Undocumented ====
#     + GetInternalTemp
#       GetTIFFInfo
#     + GetCurSeqImage
#       GetImageSize
#
#     ==== Notes ====
#     The BNS documentation states that int Board is a 1-based index, but it
#     would appear to be 0-based:  if I address board 1 with Board=1, I get an msc
#     error; using Board=0 seems to work just fine.
#     """
#
#     def __init__(self):
#         # Must chdir to module path or DLL can not find its dependencies.
#         try:
#             modpath = os.path.dirname(__file__)
#             os.chdir(modpath)
#         except:
#             # Probably running from interactive shell
#             modpath = ''
#         # path to dll
#         self.libPath = os.path.join(modpath, "PCIe16Interface")
#         # loaded library instance
#         # Now loaded here so that read_tiff is accessible even if there is no
#         # SLM present.
#         self.lib = ctypes.WinDLL(self.libPath)
#         # Boolean showing initialization status.
#         self.haveSLM = False
#         # Data type to store images.
#         self.image_size = None
#
#     ## === DECORATORS === #
#     # decorator definition for methods that require an SLM
#     def requires_slm(func):
#         def wrapper(self, *args, **kwargs):
#             if self.haveSLM == False:
#                 raise Exception("SLM is not initialized.")
#             else:
#                 return func(self, *args, **kwargs)
#
#         return wrapper
#
#     ## === PROPERTIES === #
#     @property
#     @requires_slm
#     def curr_seq_image(self):  # tested - works
#         return self.lib.GetCurSeqImage(c_int(0))
#
#     @property
#     @requires_slm
#     def power(self):  # tested - works
#         return self.lib.GetSLMPower(c_int(0))
#
#     @power.setter
#     @requires_slm
#     def power(self, value):  # tested - works
#         self.lib.SLMPower(c_int(0), c_bool(value))
#
#     @property
#     @requires_slm
#     def temperature(self):  # tested - works
#         return self.lib.GetInternalTemp(c_int(0))
#
#     ## === METHODS === #
#
#     ## Don't call this unless an SLM was initialised:  if you do, the next call
#     # can open a dialog box from some other library down the chain.
#     @requires_slm
#     def cleanup(self):  # tested
#         try:
#             self.lib.Deconstructor()
#         except:
#             pass
#         self.haveSLM = False
#
#     def initialize(self):  # tested
#         ## Need to unload and reload the DLL here.
#         # Otherwise, the DLL can open an error window about having already
#         # initialized another DLL, which we won't see on a remote machine.
#         if self.lib:
#             while (windll.kernel32.FreeLibrary(self.lib._handle)):
#                 # Keep calling FreeLibrary until library is really closed.
#                 pass
#         try:
#             # re-open the DLL
#             self.lib = ctypes.WinDLL(self.libPath)
#         except:
#             raise
#
#         # Initlialize the library, looking for nematic SLMs.
#         n = self.lib.Constructor(c_int(1))
#         if n == 0:
#             raise Exception("No SLM device found.")
#         elif n > 1:
#             raise Exception("More than one SLM device found. This module " \
#                             "can only handle one device.")
#         else:
#             self.haveSLM = True
#             self.size = self.lib.GetImageSize(0)
#             self.image_size = bnsdatatype * (self.size * self.size)
#         # SLM shows nothing without calibration, so set flat WFC.
#         white = self.image_size(65535)
#         self.write_cal(1, white)
#
#     @requires_slm
#     def load_lut(self, filename):  # tested - no errors
#         ## Warning: opens a dialog if it can't read the LUT file.
#         # Should probably check if the LUT file exists and validate it
#         # before calling LoadLUTFile.
#         self.lib.LoadLUTFile(c_int(0), c_char_p(filename))

    # @requires_slm
    # def load_sequence(self, imageList):  # tested - no errors
    #     # imageList is a list of images, each of which is a list of integers.
    #     if len(imageList) < 2:
    #         raise Exception("load_sequence expects a list of two or more " \
    #                         "images - it was passed %s images."
    #                         % len(imageList))
    #
    #     if all([type(image) is self.image_size for image in imageList]):
    #         # Data is fine as it is.
    #         pass
    #     else:
    #         # Some images need converting.
    #         flatImages = []
    #         for image in imageList:
    #             if type(image) is np.ndarray:
    #                 flatImages.append(self.image_size(*image.flatten()))
    #             else:
    #                 flatImages.append(self.image_size(*image))
    #         imageList = flatImages
    #
    #     # Make a contiguous array.
    #     sequence = (self.image_size * len(imageList))(*imageList)
    #     # LoadSequence (int Board, unsigned short* Image, int NumberOfImages)
    #     self.lib.LoadSequence(c_int(0), ctypes.byref(sequence),
    #                           c_int(len(imageList)))
    #
    # def read_tiff(self, filePath):
    #     ## void ReadTIFF (const char* FilePath, unsigned short* ImageData,
    #     #                unsigned int ScaleWidth, unsigned int ScaleHeight)
    #     buffer = self.image_size()
    #     self.lib.ReadTIFF(c_char_p(filePath), buffer,
    #                       self.size, self.size)
    #     return buffer
    #
    # @requires_slm
    # def set_sequencing_framrate(self, frameRate):  # tested - no errors
    #     ## Note - probably requires internal-triggering DLL,
    #     # rather than that set up for external triggering.
    #     self.lib.SetSequencingRate(c_double(frameRate))
    #
    # @requires_slm
    # def set_true_frames(self, trueFrames):  # tested - no errors
    #     self.lib.SetTrueFrames(c_int(0), c_int(trueFrames))
    #

    ------------
    # @requires_slm
    # def start_sequence(self):  # tested - works
    #     self.lib.StartSequence()
    #
    # @requires_slm
    # def stop_sequence(self):  # tested - works
    #     self.lib.StopSequence()
    #
    # @requires_slm
    # def write_cal(self, type, calImage):  # tested - no errors
    #     ## void WriteCal(int Board, CAL_TYPE Caltype={WFC;NUC},
    #     #               unsigned char* Image)
    #
    #     ## Not sure what type to pass the  CAL_TYPE as ... this is an ENUM, so
    #     # could be compiler / platform dependent.
    #     # 1 = WFC = wavefront correction
    #     # 0 = NUC = non-uniformity correction.
    #
    #     ## Image is a 1D array containing values from the 2D image.
    #     # image = (c_char * len(calImage))(*calImage)
    #     # Doesn't seem to like c_char, which doesn't make sense anyway, as
    #     # the calibration files are 16-bit.
    #     # Header file states it's an unsigned short.
    #
    #     image = self.image_size(*calImage)
    #     self.lib.WriteCal(c_int(0), c_int(type), ctypes.byref(image))
    #
    # @requires_slm
    # def write_image(self, image):  # tested - works
    #     ## void WriteImage (int Board, unsigned short* Image)
    #     if type(image) is self.image_size:
    #         self.lib.WriteImage(c_int(0),
    #                             ctypes.byref(self.image_size(*image)))
    #     elif type(image) is np.ndarray:
    #         self.lib.WriteImage(0, self.image_size(*image.flatten()))
    #     else:
    #         raise Exception('Unable to convert image.')
