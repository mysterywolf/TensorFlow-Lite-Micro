/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/system_setup.h"

#include <limits>

#include "tensorflow/lite/micro/debug_log.h"

#if defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)
//#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)

#ifndef ARDUINO_EXCLUDE_CODE

#include "Arduino.h"

#define DEBUG_SERIAL_OBJECT (Serial)

namespace tflite {

constexpr unsigned long kSerialMaxInitWait = 4000;  // milliseconds

void InitializeTarget() {
  DEBUG_SERIAL_OBJECT.begin();
  unsigned long start_time = millis();
  while (!DEBUG_SERIAL_OBJECT) {
    // allow for Arduino IDE Serial Monitor synchronization
    if (millis() - start_time > kSerialMaxInitWait) {
      break;
    }
  }
}

}  // namespace tflite

#endif  // ARDUINO_EXCLUDE_CODE
