from collections import deque
from collections import defaultdict
import argparse
import supervision as sv
import numpy as np
import cv2
from inference import get_model

#SOURCE = np.array([[0,300],[1000,300],[1280,440],[1280,720],[0,720]]) #polygon size
SOURCE = np.array([[0,300],[1000,300],[1800,720],[0,720]]) #polygon size
TARGET_WIDTH = 25 # this is based of real measurements of the road
TARGET_HEIGHT = 50 # this is also based of real measurements but here its a guess value
# in real life scenarios we need the real measurements of the road to provide accurate measurements of the speed
TARGET = np.array([
    [0,0],
    [TARGET_WIDTH-1,0],
    [TARGET_WIDTH-1, TARGET_HEIGHT-1],
    [0,TARGET_HEIGHT-1]
])

coordinates = defaultdict(lambda: deque(maxlen=video_info.fps)) #set up a deque

class viewTransformer:
    def __init__(self,source:np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source,target)

    def transform_points(self,points:np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1,1,2).astype(np.float32) # open cv expects points to be in 3d plane, so reshape to 3d plane
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape( -1,2)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Detection"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the video file",
        type=str,
    )
    return parser.parse_args()

if __name__=="__main__":
    args = parse_arguments()
    model = get_model(model_id="yolov8n-640")

    video_info  = sv.VideoInfo.from_video_path(args.source)
    byte_track = sv.ByteTrack(frame_rate=video_info.fps) # for speed calculation

    thickness = sv.calculate_optimal_line_thickness(resolution_wh = video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh = video_info.resolution_wh)

    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness,text_position=sv.Position.TOP_CENTER, color_lookup=sv.ColorLookup.TRACK )
    trace_annotator = sv.TraceAnnotator(thickness=thickness,trace_length=video_info.fps*2, position=sv.Position.BOTTOM_CENTER,color_lookup=sv.ColorLookup.TRACK)

    polygon_zone = sv.PolygonZone(SOURCE) #only identify vehicles inside this polygon

    frame_generator = sv.get_video_frames_generator(args.source)

    view_transformer = viewTransformer(source=SOURCE, target=TARGET) 

    for frame in frame_generator:
        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections[polygon_zone.trigger(detections=detections)]
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int) #get x,y coordinates


        direction = "in"


        labels = []
        for tracker_id,[_,y] in zip(detections.tracker_id,points):
            coordinates[tracker_id].append(y)
            if(len(coordinates[tracker_id])<video_info.fps/2):
                labels.append(f"#{tracker_id}")
            else:
                coordinates_start = coordinates[tracker_id][-1]
                coordinates_end = coordinates[tracker_id][0]
                if coordinates_start>coordinates_end:
                    direction = "in"
                else:
                    direction = "out"
                distance = abs(coordinates_start -  coordinates_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed  = distance / time * 3.6
                labels.append(f"#{tracker_id} {direction} {int(speed)} km/h")



        annotated_frame = frame.copy()
        ##annotated_frame = sv.draw_polygon(annotated_frame,polygon=SOURCE, color= sv.Color.RED) # draw a polygon for visualisation
        annotated_frame = bounding_box_annotator.annotate(scene = annotated_frame , detections = detections)
        annotated_frame = label_annotator.annotate(scene = annotated_frame , detections = detections, labels=labels)
        annotated_frame = trace_annotator.annotate(scene=annotated_frame,detections=detections)

        cv2.imshow("Annotated Frame", annotated_frame)
        if cv2.waitKey(1)==ord('q'):
            break
    cv2.destroyAllWindows()


