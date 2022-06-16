# Part 1

### Q1 - In the lab, you used the Ndida sink nv3dsink; Nvidia provides a another sink, nveglglessink. Convert the following sink to use nveglglessink.
> `gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! videoscale ! nveglglessink`

### Q2.1 - What is the difference between a property and a capability?
> Properties are names attributes of an element. These properties can be changed to alter the behavior of the element.
> Capabilities are what a pad or template is capable of doing. The difference between the two is that properties are mutable characteristics that can be changed with different parameters. Whereas the capabilities are immutable characteristics which cannot be changed with different parameters. However, sometimes an element may have many capabilities which can be used without issue and the developer can specify explicitly which capabilities should be used in the pipeline. 

### Q2.2 - How are they each expressed in a pipeline?
> Properties are set as space separated key/value pairs with an equal sign. Example: `PropertyName=SomeValue`. Capabilities are expressed as comma separated values following the element. 

### Q3.1 - Explain the following pipeline, that is explain each piece of the pipeline, describing if it is an element (if so, what type), property, or capability.
#### `gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw, framerate=30/1 ! videoconvert ! agingtv scratch-lines=10 ! videoconvert ! xvimagesink sync=false`

- `gst-launch-1.0`
    - > This is not an element, property, or capability. It is the command-line tool to build and run GStream pipelines.
- `v4l2src`
    - > This is a source element. It captures the video stream from the webcam attached to the Jetson.
- `device=/dev/video0`
    - > This is a property which specifies which device the `v4l2src` should connect and get the data from. 
- `video/x-raw`
    - > This is a capability which specifies the video format the video shall be. 
- `framerate=30/1`
    - > This is a property specifying the frame rate of the video.
- `videoconvert`
    - > This is a converting element which converts the video format. 
- `agingtv`
    - > This is a converter element which adds visual artificats to the video stream much like an old-timey TV. 
- `scratch-lines=10`
    - > A property specifying how many vertical line artifacts the `agingtv` element should add to the video feed. 
- `videoconvert`
    - > Another converter element to convert the video feed to a compatible format for the next element. 
- `xvimagesink`
    - > The sink element which renders the video feed to a display. 
- `sync=false`
    - > A property which controls whether the video display should be ran in synchronous mode with the monitor. 


### Q3.2 - What does this pipeline do?
> This pipeline gets the video feed from the monitor, adds some old-timey visual artifacts, and displays the resultant video to the display. 

### Build Python using OpenCV to capture Gstreamer Video
See `pipeline.py`. Although this task confused me a little as the example code seems to already meet all the requirements. 


# Part 2
### Base Model Used
> Resnet-18

### Description of the Dataset
> I used the Kaggle dataset [Fruits360](https://www.kaggle.com/datasets/moltean/fruits) which has 90,380 images of 131 fruits and vegetables. Frankly, I am not very impressed with this dataset. Although there are lots of pictures, the images for each fruit seem to be the exact same specimen pictured multiple times from different angles under the exact same conditions. The backgrounds are also cropped and replaced with a white border. I don't see this model generalizing well to real-time inference because the data is so sterile and unvarying. Even the test data seems to be the exact same object specimen just pictured from different angles under exact conditions. 

### How long you trained your model, how many epochs you specified, and the batch size.
> I used the default parameters for training the model via the `jetson-inference` docker image's python classification code. This means I trained my model for about 5 hours and it took only 3 epochs to reach a validation accuracy of 87.5%. I stopped at 3 epochs after I saw the model did not improve accuracy much from the previous epoch. The batch size was the default value of 8 and the default workers was 2. 

### Native Pytorch baseline


### TensorRT performance numbers
