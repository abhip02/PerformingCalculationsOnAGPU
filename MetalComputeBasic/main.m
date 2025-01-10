/*
See LICENSE folder for this sample’s licensing information.

Abstract:
An app that performs a simple calculation on a GPU.
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "MetalCompute.h"


// adding in more functionality


int main(int argc, const char * argv[]) {
    @autoreleasepool {
        
        // find a GPU
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        // Create the custom object used to encapsulate the Metal code.
        // Initializes objects to communicate with the GPU.
        MetalCalc* calc = [[MetalCalc alloc] initWithDevice:device];
        
        // Create buffers to hold data (add/sub)
        [calc prepareData];
        
        // Create buffers to hold data (dot)
        [calc prepareDataDot];
        
        // Create buffers to hold data (dot)
        [calc prepareDatapSum];
        
        // Create buffers to hold data (matmul)
        [calc prepareDataMatMul];
        
        // Send a command to the GPU to perform the calculation.
        [calc sendComputeCommand];

        NSLog(@"Execution finished");
    }
    return 0;
}
