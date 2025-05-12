import os
import cv2
import numpy as np
import threading
import time
from datetime import datetime, timedelta
from flask import Flask
from flask_cors import CORS
from ultralytics import YOLO
import requests
import queue
import logging
from threading import Lock
from collections import deque
import json
import sys

# Create a simple debug log file that will capture key information
debug_log_file = open("tracking_debug.txt", "w")

def debug_print(message):
    """Print to both console and debug file"""
    print(message)
    debug_log_file.write(f"{message}\n")
    debug_log_file.flush()  # Ensure it's written immediately

debug_print(f"=== TRACKING DEBUG LOG ===")
debug_print(f"Started at: {datetime.now()}")
debug_print(f"OpenCV version: {cv2.__version__}")

# Set up basic logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config["APPLICATION_ROOT"] = "/office-ai"

# Model configuration with performance settings
try:
    debug_print("Loading YOLO model...")
    model = YOLO("best.pt", verbose=False)
    # Set higher confidence threshold for faster detection
    model.conf = 0.3  # Only detect people with 30% confidence or higher
    model.iou = 0.45  # Higher IoU for better distinction between people
    debug_print("YOLO model loaded successfully")
except Exception as e:
    debug_print(f"Failed to load YOLO model: {str(e)}")
    model = None

# Tracking variables
TRACK_HISTORY = 10  # Shortened for faster response with fast-moving people
frame_queue = queue.Queue(maxsize=10)
stats_lock = Lock()

class PeopleTracker:
    def __init__(self):
        self.names = model.names if model else {}
        self.active_people = 0
        self.entered_zone = 0
        self.exited_zone = 0
        
        # Track history for each track_id
        self.track_history = {}     # track_id -> list of points
        self.track_states = {}      # track_id -> state dictionary
        
        # Count tracking
        self.total_entries = 0      # Total entry count
        self.total_exits = 0        # Total exit count
        
        # Appearance/disappearance tracking
        self.appearing_tracks = {}  # Tracks that just appeared
        self.disappearing_tracks = {}  # Tracks about to disappear
        self.disappeared_locations = {}  # Last locations of disappeared tracks
        
        # Frame counter and processing speed tracking
        self.frame_counter = 0
        self.processing_times = deque(maxlen=30)  # Store last 30 processing times
        
        # Define zones with clear borders matching the room layout
        self.areas = {
            # ENTRY zone (green) - upper part
            'entry': np.array([(327, 250), (322, 328), (880, 328), (880, 250)], np.int32),
            
            # EXIT zone (red) - lower part
            'exit': np.array([(322, 336), (312, 420), (880, 420), (880, 336)], np.int32),
            
            # Door area - near the top of the frame
            'door': np.array([(200, 50), (200, 200), (880, 200), (880, 50)], np.int32),
            
            # Bottom area - for appearance detection
            'bottom': np.array([(0, 450), (0, 600), (1020, 600), (1020, 450)], np.int32)
        }
        
        # Motion detection thresholds - optimized for speed
        self.MIN_TRACK_POINTS = 2   # Minimum points needed (reduced for fast movement)
        self.DIRECTION_THRESHOLD = 3  # Smaller threshold to detect fast movements
        self.DISAPPEARANCE_TIMEOUT = 10  # Frames to wait before counting a disappearance
        
        # Speed and position thresholds
        self.MIN_SPEED_PPS = 2.0    # Minimum pixels per frame to be considered "moving"
        
        # Employee tracking
        self.employee_types = {}    # track_id -> employee type (P1 or P2)
        self.logged_entries = set() # Set of track IDs that have had entries logged
        self.logged_exits = set()   # Set of track IDs that have had exits logged
        
        debug_print("PeopleTracker initialized with the following settings:")
        debug_print(f"- DIRECTION_THRESHOLD: {self.DIRECTION_THRESHOLD}")
        debug_print(f"- MIN_TRACK_POINTS: {self.MIN_TRACK_POINTS}")
        debug_print(f"- TRACK_HISTORY: {TRACK_HISTORY}")
        debug_print(f"- Entry zone: {self.areas['entry'].tolist()}")
        debug_print(f"- Exit zone: {self.areas['exit'].tolist()}")
        debug_print(f"- Door zone: {self.areas['door'].tolist()}")
        debug_print(f"- Bottom zone: {self.areas['bottom'].tolist()}")

    def assign_employee_type(self, track_id, class_name="Person"):
        """Assign employee type based on class detection (from old logic)"""
        if "P1" in class_name:
            return 'P1'
        elif "P2" in class_name:
            return 'P2'
        else:
            # Fallback to original assignment if not explicitly P1 or P2
            return 'P1' if track_id % 2 == 0 else 'P2'

    def log_employee_entry(self, employee_type):
        """Log employee entry to API"""
        emp_id = 31 if employee_type == 'P1' else 32
        self.send_log(emp_id, 1)

    def log_employee_exit(self, employee_type):
        """Log employee exit to API"""
        emp_id = 31 if employee_type == 'P1' else 32
        self.send_log(emp_id, 2)

    def send_log(self, employee_id, log_type):
        """Send log to API with retry logic"""
        adjusted_time = datetime.now() + timedelta(hours=3)
        current_time = adjusted_time.strftime("%Y-%m-%d %H:%M:%S")
        api_url = "https://backai.mycashtest.com/apiAdmin/employee/create_log"
        params = {
            "employee_id": employee_id,
            "type": log_type,
            "date": current_time
        }
        
        # Log the request being sent
        debug_print(f"[API] Sending employee log: ID={employee_id}, type={log_type}, time={current_time}")
        
        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                response = requests.post(api_url, params=params, timeout=3)
                response.raise_for_status()  # Raise error for bad status codes
                logger.info(f"Log added for employee {employee_id}, type {log_type}")
                debug_print(f"[API] Log successfully added for employee {employee_id}, type {log_type}")
                return
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                debug_print(f"[API] Attempt {attempt+1} failed: {str(e)}")
                if attempt < retry_attempts - 1:
                    logger.info(f"Retrying in 2 seconds...")
                    debug_print(f"[API] Retrying in 2 seconds...")
                    time.sleep(2)  # Wait before retrying
        
        # If we get here, all retry attempts failed
        debug_print(f"[API] All retry attempts failed for employee {employee_id}, type {log_type}")

    def calculate_speed(self, track_id):
        """Calculate the speed of a track in pixels per frame"""
        history = self.track_history.get(track_id, [])
        if len(history) < 2:
            return 0.0
            
        # Calculate displacement between the last two points
        last = history[-1]
        prev = history[-2]
        dx = last[0] - prev[0]
        dy = last[1] - prev[1]
        
        # Return Euclidean distance (speed magnitude)
        return np.sqrt(dx*dx + dy*dy)

    def handle_person_movement(self, track_id, point, class_name="Person"):
        test_point = (int(point[0]), int(point[1]))
        
        # Check if this is a new track
        is_new_track = track_id not in self.track_history
        
        # Initialize track history and state if not exists
        if is_new_track:
            self.track_history[track_id] = []
            self.track_states[track_id] = {
                'direction': None,
                'in_entry_zone': False,
                'in_exit_zone': False,
                'in_door_zone': False,
                'in_bottom_zone': False,
                'last_entry_count': 0,  # Frame when last entry was counted
                'last_exit_count': 0,   # Frame when last exit was counted
                'appeared_in_bottom': False,  # Flag if appeared in bottom zone
                'first_seen_frame': self.frame_counter
            }
            self.appearing_tracks[track_id] = self.frame_counter
            
            # Assign employee type based on class detection from old logic
            employee_type = self.assign_employee_type(track_id, class_name)
            self.employee_types[track_id] = employee_type
            debug_print(f"[NEW] Track {track_id} assigned employee type: {employee_type} based on class: {class_name}")
            
            # Check if this is appearing in the bottom zone
            in_bottom = cv2.pointPolygonTest(self.areas['bottom'], test_point, False) >= 0
            if in_bottom:
                self.track_states[track_id]['appeared_in_bottom'] = True
                debug_print(f"[APPEAR] Track {track_id} appeared in BOTTOM zone at {test_point}")
                
                # Only count as entry if they weren't already counted in the entry zone
                if track_id not in self.logged_entries:  # Check if they were counted via entry zone
                    self.total_entries += 1
                    self.active_people += 1
                    self.logged_entries.add(track_id)
                    debug_print(f"[COUNT-APPEAR] Person {track_id} ENTERED (appeared at bottom). Total entries: {self.total_entries}, Active: {self.active_people}")
                    
                    # Log employee entry
                    emp_type = self.employee_types.get(track_id, 'P1')  # Default to P1 if not set
                    self.log_employee_entry(emp_type)
            else:
                debug_print(f"[NEW] Track {track_id} at position {test_point}")
        
        # Add point to track history
        self.track_history[track_id].append(test_point)
        
        # Limit history length
        if len(self.track_history[track_id]) > TRACK_HISTORY:
            self.track_history[track_id].pop(0)
        
        # Get state for this track
        state = self.track_states[track_id]
        
        # Determine current position relative to zones
        in_entry = cv2.pointPolygonTest(self.areas['entry'], test_point, False) >= 0
        in_exit = cv2.pointPolygonTest(self.areas['exit'], test_point, False) >= 0
        in_door = cv2.pointPolygonTest(self.areas['door'], test_point, False) >= 0
        in_bottom = cv2.pointPolygonTest(self.areas['bottom'], test_point, False) >= 0
        
        # Track zone status changes
        zone_changed = False
        if in_entry != state['in_entry_zone']:
            state['in_entry_zone'] = in_entry
            zone_changed = True
            if in_entry:
                debug_print(f"[ZONE] Track {track_id} entered ENTRY zone at {test_point}")
            else:
                debug_print(f"[ZONE] Track {track_id} left ENTRY zone at {test_point}")
        
        if in_exit != state['in_exit_zone']:
            state['in_exit_zone'] = in_exit
            zone_changed = True
            if in_exit:
                debug_print(f"[ZONE] Track {track_id} entered EXIT zone at {test_point}")
            else:
                debug_print(f"[ZONE] Track {track_id} left EXIT zone at {test_point}")
                
        if in_door != state['in_door_zone']:
            state['in_door_zone'] = in_door
            zone_changed = True
            if in_door:
                debug_print(f"[ZONE] Track {track_id} entered DOOR zone at {test_point}")
            else:
                debug_print(f"[ZONE] Track {track_id} left DOOR zone at {test_point}")
                
        if in_bottom != state['in_bottom_zone']:
            state['in_bottom_zone'] = in_bottom
            zone_changed = True
            if in_bottom:
                debug_print(f"[ZONE] Track {track_id} entered BOTTOM zone at {test_point}")
            else:
                debug_print(f"[ZONE] Track {track_id} left BOTTOM zone at {test_point}")
        
        # Calculate direction
        current_y = test_point[1]
        history_length = len(self.track_history[track_id])
        old_direction = state['direction']  # Save previous direction for logging
        
        # Update direction if we have enough points
        if history_length >= self.MIN_TRACK_POINTS:
            # Use the oldest point to determine overall direction
            first_y = self.track_history[track_id][0][1]
            y_diff = current_y - first_y
            
            # Only update direction if the change is significant
            if abs(y_diff) > self.DIRECTION_THRESHOLD:
                new_direction = 'down' if y_diff > 0 else 'up'
                
                # If direction changed, log it
                if new_direction != old_direction:
                    debug_print(f"[DIR] Track {track_id} direction changed: {old_direction} -> {new_direction} (y_diff: {y_diff})")
                
                state['direction'] = new_direction
                
        # Calculate speed
        speed = self.calculate_speed(track_id)
        
        # If track was in door zone and has disappeared, remove from appearing tracks
        if track_id in self.appearing_tracks and not in_door:
            del self.appearing_tracks[track_id]
        
        # COUNTING LOGIC - multiple scenarios
        # 1. PRIMARY: Zone-based detection
        cooldown_frames = 5  # Prevent counting too frequently
        
        # Entry counting: person in entry zone moving downward
        if (in_entry and 
            state['direction'] == 'down' and 
            self.frame_counter - state['last_entry_count'] > cooldown_frames and
            speed > self.MIN_SPEED_PPS and  # Must be moving
            track_id not in self.logged_entries):  # Check if not already logged
            
            # Count entry
            self.total_entries += 1
            self.active_people += 1
            state['last_entry_count'] = self.frame_counter
            self.logged_entries.add(track_id)  # Mark as logged
            
            debug_print(f"[COUNT] Person {track_id} ENTERED via entry zone. Total entries: {self.total_entries}, Active: {self.active_people}")
            
            # Log employee entry
            emp_type = self.employee_types.get(track_id, 'P1')  # Default to P1 if not set
            self.log_employee_entry(emp_type)
        
        # Exit counting: person in exit zone moving upward
        if (in_exit and 
            state['direction'] == 'up' and 
            self.frame_counter - state['last_exit_count'] > cooldown_frames and
            speed > self.MIN_SPEED_PPS and  # Must be moving
            track_id not in self.logged_exits):  # Check if not already logged
            
            # Count exit
            self.total_exits += 1
            self.active_people = max(0, self.active_people - 1)
            state['last_exit_count'] = self.frame_counter
            self.logged_exits.add(track_id)  # Mark as logged
            
            debug_print(f"[COUNT] Person {track_id} EXITED via exit zone. Total exits: {self.total_exits}, Active: {self.active_people}")
            
            # Log employee exit
            emp_type = self.employee_types.get(track_id, 'P1')  # Default to P1 if not set
            self.log_employee_exit(emp_type)
        
        # 2. Track door disappearances - mark as potential exit
        if in_door and track_id not in self.disappearing_tracks:
            self.disappearing_tracks[track_id] = {
                'first_seen': self.frame_counter,
                'last_position': test_point,
                'first_position': test_point,
                'in_door_frames': 1
            }
        elif track_id in self.disappearing_tracks:
            self.disappearing_tracks[track_id]['last_position'] = test_point
            if in_door:
                self.disappearing_tracks[track_id]['in_door_frames'] += 1
            else:
                # If moved away from door zone, remove from disappearing tracks
                del self.disappearing_tracks[track_id]
        
        return {
            "track_id": track_id,
            "position": test_point,
            "in_entry": in_entry,
            "in_exit": in_exit,
            "in_door": in_door,
            "in_bottom": in_bottom,
            "direction": state['direction'],
            "speed": speed,
            "employee_type": self.employee_types.get(track_id, 'P1')  # Include employee type
        }

    def check_disappeared_tracks(self, current_tracks):
        """Handle track disappearances, especially near the door"""
        # Check for disappeared tracks
        for track_id in list(self.disappearing_tracks.keys()):
            if track_id not in current_tracks:
                tracking_info = self.disappearing_tracks[track_id]
                last_pos = tracking_info['last_position']
                door_frames = tracking_info.get('in_door_frames', 0)
                
                # Only count if they spent enough time in the door zone
                if door_frames >= 3 and track_id not in self.logged_exits:  # Check if not already logged
                    # Only count as exit if they weren't already counted via exit zone
                    self.total_exits += 1
                    self.active_people = max(0, self.active_people - 1)
                    self.logged_exits.add(track_id)  # Mark as logged
                    debug_print(f"[COUNT-DISAPPEAR] Person {track_id} EXITED (disappeared at door). Total exits: {self.total_exits}, Active: {self.active_people}")
                    
                    # Log employee exit
                    emp_type = self.employee_types.get(track_id, 'P1')  # Default to P1 if not set
                    self.log_employee_exit(emp_type)
                else:
                    debug_print(f"[INFO] Person {track_id} disappeared at door but was already counted as exited or spent insufficient time in door zone")
                    
                # Track last position for possible debug visualization
                self.disappeared_locations[track_id] = {
                    'position': last_pos,
                    'frame': self.frame_counter,
                    'counted_exit': door_frames >= 3 and track_id not in self.logged_exits
                }
                
                # Remove from tracking
                del self.disappearing_tracks[track_id]
    
    def cleanup_tracks(self, current_tracks):
        # First check for disappeared tracks
        self.check_disappeared_tracks(current_tracks)
        
        # Remove inactive tracks
        inactive_tracks = []
        for track_id in list(self.track_history.keys()):
            if track_id not in current_tracks:
                inactive_tracks.append(track_id)
        
        # Remove history for inactive tracks
        for track_id in inactive_tracks:
            if track_id in self.track_history:
                if track_id in self.track_states:
                    state = self.track_states[track_id]
                    debug_print(f"[END] Track {track_id} removed. Final state: " +
                              f"direction={state['direction']}, " +
                              f"in_entry_zone={state['in_entry_zone']}, " +
                              f"in_exit_zone={state['in_exit_zone']}, " +
                              f"in_door_zone={state['in_door_zone']}, " +
                              f"employee_type={self.employee_types.get(track_id, 'Unknown')}")
                
                del self.track_history[track_id]
                
            if track_id in self.track_states:
                del self.track_states[track_id]
                
            # Remove from employee type tracking
            if track_id in self.employee_types:
                del self.employee_types[track_id]
        
        # Clean up old disappeared locations (after 50 frames)
        for track_id in list(self.disappeared_locations.keys()):
            if self.frame_counter - self.disappeared_locations[track_id]['frame'] > 50:
                del self.disappeared_locations[track_id]
        
    def update_statistics(self):
        with stats_lock:
            self.entered_zone = self.total_entries
            self.exited_zone = self.total_exits
            
            # Log every 10 frames for performance
            if self.frame_counter % 10 == 0:
                debug_print(f"[STATS] Active={self.active_people}, Entered={self.entered_zone}, Exited={self.exited_zone}")
            
    def process_frame(self, frame, frame_count, frame_skip):
        try: 
            # Start timing the processing
            start_time = time.time()
            
            self.frame_counter += 1
            
            if self.frame_counter % 30 == 0:  # Log status every 30 frames
                active_tracks = list(self.track_history.keys())
                debug_print(f"[FRAME {self.frame_counter}] Active tracks: {active_tracks}")
                debug_print(f"[STATUS] Active={self.active_people}, Entered={self.entered_zone}, Exited={self.exited_zone}")
                
                # Calculate average processing time
                if self.processing_times:
                    avg_time = sum(self.processing_times) / len(self.processing_times)
                    avg_fps = 1.0 / avg_time if avg_time > 0 else 0
                    debug_print(f"[PERFORMANCE] Avg processing time: {avg_time*1000:.1f}ms, Avg FPS: {avg_fps:.1f}")
            
            frame = cv2.resize(frame, (1020, 600))  # Resize frame
            if model is None:
                debug_print("YOLO model not loaded, cannot process frame")
                return
            
            # Run detection with performance settings
            results = model.track(frame, persist=True)  # Get model results
    
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else None
                confidences = results[0].boxes.conf.cpu().numpy()
    
                current_tracks = set()
    
                for idx, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confidences)):
                    track_id = track_ids[idx] if track_ids is not None else idx
                    c = self.names[int(class_id)]
                    x1, y1, x2, y2 = box.astype(int)
                    centered_x, centered_y = (x1 + x2) // 2, (y1 + y2) // 2
                    point = (centered_x, centered_y)
                    current_tracks.add(track_id)
    
                    color = (0, 255, 0) if "Person" in c else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
                    # Handling person tracking
                    if "Person" in c or "P1" in c or "P2" in c:
                        movement_info = self.handle_person_movement(track_id, point, c)
                        
                        # Add ID and employee type label
                        emp_type = movement_info.get("employee_type", "P1")
                        label = f"{c} {track_id} ({emp_type})"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        # Add direction indicator
                        if movement_info["direction"]:
                            direction_text = "DOWN" if movement_info["direction"] == "down" else "UP"
                            # Add speed to the label
                            speed_text = f"{movement_info['speed']:.1f}px/f"
                            cv2.putText(frame, f"{direction_text} {speed_text}", (x1 + 10, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            
                        # Draw a dot at the center point
                        cv2.circle(frame, point, 5, (0, 0, 255), -1)
                            
                    # Card detection
                    elif "Card" in c:
                        # Add a label
                        cv2.putText(frame, f"{c} {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
                self.cleanup_tracks(current_tracks)
                self.update_statistics()
    
                # Overlay statistics
                cv2.putText(frame, f"Active People: {self.active_people}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Total Entered: {self.entered_zone}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Total Exited: {self.exited_zone}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"Frame: {self.frame_counter}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
                # Get current timestamp
                timestamp = datetime.now().strftime("%m-%d-%Y %a. %H:%M:%S")
                cv2.putText(frame, timestamp, (10, 190), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
                # Draw zones with semi-transparency
                overlay = frame.copy()
                
                # Entry zone (green)
                cv2.fillPoly(overlay, [self.areas['entry']], (0, 255, 0, 128))
                cv2.polylines(frame, [self.areas['entry']], True, (0, 255, 0), 2)
                cv2.putText(frame, "ENTRY", tuple(self.areas['entry'][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Exit zone (red)
                cv2.fillPoly(overlay, [self.areas['exit']], (0, 0, 255, 128))
                cv2.polylines(frame, [self.areas['exit']], True, (0, 0, 255), 2)
                cv2.putText(frame, "EXIT", tuple(self.areas['exit'][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Door zone (blue)
                cv2.fillPoly(overlay, [self.areas['door']], (255, 0, 0, 128))
                cv2.polylines(frame, [self.areas['door']], True, (255, 0, 0), 1)
                cv2.putText(frame, "DOOR", tuple(self.areas['door'][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # Bottom zone (yellow)
                cv2.fillPoly(overlay, [self.areas['bottom']], (0, 255, 255, 128))
                cv2.polylines(frame, [self.areas['bottom']], True, (0, 255, 255), 1)
                cv2.putText(frame, "BOTTOM", tuple(self.areas['bottom'][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # Apply transparency
                alpha = 0.3
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
                # Display the frame in a window
                #cv2.imshow("Tracking Visualization", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Press 'q' to quit
                    raise KeyboardInterrupt("User stopped visualization")
                elif key == ord('d'):  # Press 'd' to dump current state
                    debug_print("\n=== CURRENT STATE DUMP ===")
                    for track_id in self.track_history:
                        if track_id in self.track_states:
                            state = self.track_states[track_id]
                            emp_type = self.employee_types.get(track_id, 'Unknown')
                            debug_print(f"Track {track_id} ({emp_type}): " +
                                      f"direction={state['direction']}, " +
                                      f"in_entry_zone={state['in_entry_zone']}, " +
                                      f"in_exit_zone={state['in_exit_zone']}, " +
                                      f"in_door_zone={state['in_door_zone']}")
                    debug_print("=======================\n")
            
            # Track processing time
            end_time = time.time()
            processing_time = end_time - start_time
            self.processing_times.append(processing_time)
    
        except Exception as e:
            debug_print(f"Frame processing error: {str(e)}")
            import traceback
            debug_print(traceback.format_exc())

# Initialize the tracker
tracker = PeopleTracker()

def process_frame(frame, frame_count, frame_skip):
    tracker.process_frame(frame, frame_count, frame_skip)

# Video capture function
def start_video_capture(video_source=0):
    """Start capturing video from the specified source."""
    try:
        debug_print(f"Opening video source: {video_source}")
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            debug_print(f"Cannot open video source {video_source}")
            return
        
        # Try to set higher resolution and fps for better tracking
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual camera parameters
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        debug_print(f"Camera parameters: {actual_width}x{actual_height} @ {actual_fps}fps")
        
        frame_count = 0
        frame_skip = 0  # Process every frame
        
        debug_print(f"Video capture started successfully")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                debug_print("Failed to read frame, video stream may have ended")
                break
                
            frame_count += 1
            
            # Skip frames for performance if needed
            if frame_count % (frame_skip + 1) == 0:
                # Process frame
                process_frame(frame, frame_count, frame_skip)
            
            # Break loop if 'q' pressed (handled inside process_frame)
    
    except KeyboardInterrupt:
        debug_print("Video capture stopped by user")
    except Exception as e:
        debug_print(f"Error in video capture: {str(e)}")
        import traceback
        debug_print(traceback.format_exc())
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        debug_print("Video capture resources released")
        
        # Final stats
        debug_print("\n=== FINAL STATISTICS ===")
        debug_print(f"Total frames processed: {tracker.frame_counter}")
        debug_print(f"Final counts: Active={tracker.active_people}, Entered={tracker.entered_zone}, Exited={tracker.exited_zone}")
        debug_print("=======================\n")
        
        # Close debug log
        debug_log_file.close()

if __name__ == "__main__":
    # Log system info
    debug_print(f"Starting people tracking system")
    
    try:
        # Try to create video source from camera or fallback to file
        video_source = 0  # Default to camera 0
        
        # Check if we should use a video file instead
        if len(sys.argv) > 1:
            video_source = sys.argv[1]
            debug_print(f"Using video source from command line: {video_source}")
        
        # Start video capture 
        start_video_capture(video_source)
        
    except Exception as e:
        debug_print(f"Startup error: {str(e)}")
        import traceback
        debug_print(traceback.format_exc())
    
    debug_print("Program finished. Debug log is available at: tracking_debug.txt")