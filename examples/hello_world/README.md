# Hello World Example

This example is designed to demonstrate the absolute basics of using [TensorFlow
Lite for Microcontrollers](https://github.com/RTduino-libraries/tflite-micro/tree/tinyml-petebook-baseline/tensorflow/lite/micro/examples/hello_world).
It includes the full end-to-end workflow of training a model, converting it for
use with TensorFlow Lite for Microcontrollers for running inference on a
microcontroller.

The model is trained to replicate a `sine` function and generates a pattern of
data to blink the built-in LED in a fade in/out pattern.

## Deploy to Arduino

The following instructions will help you build and deploy this sample
to [Arduino](https://www.arduino.cc/) devices.

The sample has been tested with the following devices:

- [RTduino STM32F412 Nulceo Board](https://github.com/RT-Thread/rt-thread/tree/master/bsp/stm32/stm32f412-st-nucleo)

The sample will use PWM to fade an LED on and off according to the model's output. In the code, the `LED_BUILTIN` constant is used to specify the board's built-in LED as the one being controlled. However, on some boards, this built-in LED is not attached to a pin with PWM capabilities. In this case, the LED will blink instead of fading.

You can modify the LED pin number in `arduino_output_handler.cpp` file to adjust to a pin which can generate the PWM. By default, it is `int led = LED_BUILTIN`. For most of the board, `LED_BUILTIN` doesn't support PWM feature, you may see the LED is blinking with changing frequency.

![Animation on Nano 33 BLE Sense](images/hello_world_animation.gif)

### Install the TensorFlow Lite Micro library

In the menuconfig:

- Enable TensorFlow Lite Micro
- Enable hello world example

```Kconfig
RT-Thread online packages  --->
    Arduino libraries  --->
        Data Processing  --->
            [*] TensorFlow Lite Micro: TensorFlow Lite for Microcontrollers (TLFM) Library for RTduino --->
                Examples --->
                    [*] Enable hello world example
```

### Load and run the example

Once you select `Enable hello world example`, after compiling, the hello world example will automatically run, and you will see the LED is changing. Meanwhile, the serial also will print a sine wave time series number.

![Serial Plotter with Nano 33 BLE Sense](images/hello_world_serial_plotter.png)

![GUI Plotter with STM32F746](images/animation_on_STM32F746.gif)

### Reference

- https://www.tensorflow.org/lite/microcontrollers/get_started_low_level
- https://www.youtube.com/watch?v=BzzqYNYOcWc
- https://www.youtube.com/watch?v=8N6-WQsxwxA&list=PLtT1eAdRePYoovXJcDkV9RdabZ33H6Di0&index=2
