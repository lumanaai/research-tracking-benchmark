# Shared Memory Streaming Test

### Steps:

* You need to compile once (unless you do some changes in code). 

1. compile transmitter tester:
    ```
    $ cd gstreamer_tester/transmitter

    $ nvcc -Xcompiler -fPIC --shared -o transmitter_tester.so -O3 transmitter_tester.cu -I/TransmitterManager.h -I/usr/local/include/opencv4/ -I/usr/local/cuda/include -lrt -lopencv_core -lopencv_videoio -lopencv_highgui 
    ```


2. Run transmitter tester script, example:
    ```
    $ python3 transmitter_test.py -h

        Transmit video frames over shared memory

        optional arguments:
        -h, --help       show this help message and exit
        --src SRC        video source path
        --rate RATE      streaming framerate
        --frames FRAMES  number of frames to transmit
    
    $ python3 transmitter_test.py --src "/home/nvidia/analytics/data/garage.mp4" --rate 4 --frames 200

        timestamp = 1657075729385
    ```

3. compile receiver tester:
    ```
    $ cd gstreamer_tester/receiver

    $ nvcc -Xcompiler -fPIC --shared -o receiver_tester.so -O3 receiver_tester.cu -I/ReceiverManager.h -I/usr/local/include/opencv4/ -I/usr/local/cuda/include -lrt -lopencv_core -lopencv_videoio -lopencv_highgui 
    ```

4. Run receiver tester script, example:
    ```
    $ python3 receiver_tester.py -h

        Receive video frames from shared memory

        optional arguments:
        -h, --help            show this help message and exit
        --rate RATE           streaming framerate
        --timestamp TIMESTAMP
                                initial timestamp (get from transmitter)
        --frames FRAMES       number of frames to transmit
        --width WIDTH         frame width
        --height HEIGHT       frame height
    
    $ python3 receiver_tester.py --rate 4 --timestamp 1657075729385 --frames 200 --width 2592 --height 1520
    ```