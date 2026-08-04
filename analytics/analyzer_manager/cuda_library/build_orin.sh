echo "Copying cmake rules to /usr/src/jetson_multimedia_api/samples"
cp Rules.mk /usr/src/jetson_multimedia_api/samples

echo "Compiling dependencies"
pushd /usr/src/jetson_multimedia_api/
make
popd

echo "Copying object files"
cp /usr/src/jetson_multimedia_api/samples/common/classes/*.o ./object_files/
rm ./object_files/NvBufSurface.o

echo "Compiling NVJPG Encoder"
export CPPFLAGS="-I/usr/src/jetson_multimedia_api/include -I/usr/src/jetson_multimedia_api/include/libjpeg-8b -I/usr/src/jetson_multimedia_api/samples/common/algorithm/cuda -I/usr/src/jetson_multimedia_api/samples/common/algorithm/trt -I/usr/include -I/usr/include/libdrm -I/usr/include/python3.8"
export LDFLAGS="-lpthread -lv4l2 -lEGL -lGLESv2 -lX11 -lnvbuf_utils -lnvjpeg -lnvosd -ldrm -lpython3.8 -ldl  -lutil -lm  -Xlinker -export-dynamic -L/usr/lib/python3.8/config-3.8m-aarch64-linux-gnu -L/usr/lib/aarch64-linux-gnu/ -L/usr/lib/aarch64-linux-gnu/tegra"
/usr/local/cuda/bin/nvcc -Xcompiler -fPIC --shared -O3 $CPPFLAGS $LDFLAGS nvJpeg_encoder.cu -o nvJpeg_encoder.so ./object_files/*.o