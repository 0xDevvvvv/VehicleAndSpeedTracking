#Vehicle And Speed Tracking
Vehicle And Speed Tracking using OpenCV and SuperVision<br>


-----
## Project Overview :
- Tracks Different Types of vehicles
- Assigns colours based on tracker ID
- Traces the route
- Calculates Speed (*Note : currently the calculation is wrong due to unknown meansurements of the road*)
- Show whether the vehicle is coming or goind

------
**Imported Modules**

<code>from collections import deque
from collections import defaultdict
import argparse
import supervision as sv
import numpy as np
import cv2
from inference import get_model
</code>

----
To Run:

> python3 main.py --source "/path/to/video/file"
