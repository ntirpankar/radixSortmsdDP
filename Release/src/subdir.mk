################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/radixsortmsdDP.cu 

CPP_SRCS += \
../src/drecho.cpp 

OBJS += \
./src/drecho.o \
./src/radixsortmsdDP.o 

CU_DEPS += \
./src/radixsortmsdDP.d 

CPP_DEPS += \
./src/drecho.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -I/uufs/chpc.utah.edu/common/home/u0686941/cuda-workspace/radixsortmsdDP/cub-1.7.5 -O3 -std=c++11 -gencode arch=compute_60,code=sm_60  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -I/uufs/chpc.utah.edu/common/home/u0686941/cuda-workspace/radixsortmsdDP/cub-1.7.5 -O3 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -I/uufs/chpc.utah.edu/common/home/u0686941/cuda-workspace/radixsortmsdDP/cub-1.7.5 -O3 -std=c++11 -gencode arch=compute_60,code=sm_60  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -I/uufs/chpc.utah.edu/common/home/u0686941/cuda-workspace/radixsortmsdDP/cub-1.7.5 -O3 -std=c++11 --compile --relocatable-device-code=true -gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


