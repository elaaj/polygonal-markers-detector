from numpy import ndarray
from cv2 import (
    VideoCapture,
    VideoWriter,
    VideoWriter_fourcc,
    CAP_PROP_FRAME_COUNT,
    # imshow,
    # waitKey,
    destroyAllWindows,
)
import marker_detector as mkdtct


def detectMarkerAndTrack(objectToTrack: int) -> ndarray:
    # Initialize both video reader and writer.
    videoCapPath = f"../data/obj0{objectToTrack}.mp4"
    # ! I open the file in a write mode and if it doesn't exist it will be created.
    f = open(f"obj{objectToTrack}_marker.csv", "w")
    # ! I write a header to it.
    f.write("FRAME, MARK_ID,   Px,   Py,    X,    Y, Z\n")
    f.close()
    # ! I'm creating a VideoCapture object from the input video file specified by videoCapPath.
    vidcap = VideoCapture(videoCapPath)
    storedFrames = []  # ! initialized an empty list
    success, frame = vidcap.read()  # ! I'm reading the first frame.
    framesCount = int(
        vidcap.get(CAP_PROP_FRAME_COUNT)
    )  # ! Getting the total number of frames.
    videoFormat = VideoWriter_fourcc(
        "m", "p", "4", "v"
    )  # ! Setting the video format to "mp4v"
    videoWriterPath = (
        "../data/obj" + str(objectToTrack) + "_marker.mp4"
    )  # ! Setting the output video file path.
    # ! I'm creating the VideoWriter object using the specified video format,
    # ! frame rate and frame size.
    videoWriter = VideoWriter(
        videoWriterPath,
        videoFormat,  # Every output-video will be produced in this format.
        29.97,  # Every input-video has this frame rate.
        (1920, 1080),  # Every input-video has these shapes.
    )

    # If you want to view the video while being modified in real time,
    # uncomment the following commented lines, and also the releated
    # imports at the beginning.
    # ! This loop iterates over all frames of the input video. For each frame,
    # ! it calls the detectAndLabelMarkers() function to detect and label
    # ! markers in the frame, appends the frame to the storedFrames list, and
    # ! reads the next frame using the read() method of the VideoCapture object.
    # ! Once all frames have been processed, it releases the VideoCapture object
    # ! using the release() method.
    for index in range(0, framesCount):
        if success:
            mkdtct.detectAndLabelMarkers(
                image=frame, currentFrame=index, objectToTrack=objectToTrack
            )
            # imshow("Marker Detection and Tracking", frame)
            storedFrames.append(frame)
            # k = waitKey(30) & 0xFF
            success, frame = vidcap.read()
            # if k == 27:
            #     break
        else:
            break
    vidcap.release()

    # destroyAllWindows()

    # ! Loop over all frames stored in storedFrames list and write each frame to
    # ! the output video file using the write method. And then realese the VideoWriter object.
    _ = [videoWriter.write(frame) for frame in storedFrames]

    videoWriter.release()


if __name__ == "__main__":
    # Selectable videos
    loadableVideos = {"Toucan": 1, "Dino": 2, "Cracker": 3, "Ganesh": 4}

    # Choose one among the keys in loadableVideos
    objectToTrack = loadableVideos["Ganesh"]

    detectMarkerAndTrack(objectToTrack)
