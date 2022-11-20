import cv2
import mediapipe as mp
import sys, getopt
import time
import numpy as np

try:
	import cv2
	from ar_markers import detect_markers
except ImportError:
	raise Exception('Error: OpenCv is not installed')

information_visual = True
# information_visual = False
cam = 1 # default = 0

# class bullet():
# 	def __init__(self, pos, vec):
# 		self.pos
# 		self.life_time = 5
# 		self.t = time.time()
# 		self.vec
#
# 	def attack(self, position):
# 		 # 포지션안에 불릿 pos위치시 작동
# 		self.pos
# 		p = 1
# 		return p
#
# 	def move(self):
# 		t1 = time.time()
# 		if t1 - self.time > self.life_time:
# 			del self
# 		self.pos = (self.vec)*(t1 - self.t)
# 		if self.pos[0] < 0 or self.pos[0] > 1280 or self.pos[1] < 0 or self.pos[1] > 720:
# 			del self

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

IMAGE_FILES = []
with mp_hands.Hands(
	static_image_mode=True,
	max_num_hands=2,
	min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)




def main(argv):
	p1_hp = 100
	p2_hp = 100
	p1_pose = []
	p2_pose = []
	p1_center = []
	p2_center = []
	p1_rotation = 0
	p2_rotation = 0

	p1_id = 2495
	p2_id = 482

	play_time = 120.0
	game_play = False
	start_time = 0
	marker_num = 0
	marker_set = 0
	start_count = 3

	print('Press "q" to quit')
	capture = cv2.VideoCapture(cam)

	if capture.isOpened():  # try to get the first frame
		frame_captured = capture.read()
	else:
		print('Failed to Open Camera %d' % cam)
		frame_captured = False
	while frame_captured:
		frame_captured, frame = capture.read()

		markers = detect_markers(frame)

		# print(frame.shape) #(720, 1280,3)


		frame_size = frame.shape
		cv2.rectangle(frame, (int(frame_size[1] * 0.04), int(frame_size[0] * 0.05)), ((int(frame_size[1] * 0.40 * (p1_hp / 100) + frame_size[1] * 0.04)), int(frame_size[0] * 0.10)),
									(255, 0, 0), -1)
		cv2.rectangle(frame, (int(frame_size[1] * 0.04), int(frame_size[0] * 0.05)),
									(int(frame_size[1] * 0.44), int(frame_size[0] * 0.10)),
									(0,0,0), 3)
		cv2.rectangle(frame, (int(frame_size[1] * 0.56), int(frame_size[0] * 0.05)), (int(frame_size[1] * 0.96), int(frame_size[0] * 0.10)),
									(0, 0, 255), -1)
		cv2.rectangle(frame, (int(frame_size[1] * 0.56), int(frame_size[0] * 0.05)), ((int(frame_size[1] * 0.40 * (p2_hp / 100) + frame_size[1] * 0.56)), int(frame_size[0] * 0.10)),
									(0,0,0), 3)

		if game_play == True:
			play_time = 120 - (time.time() - start_time)
			if play_time < 0:
				play_time = 0
				game_play = False
		cv2.putText(frame, str(round(play_time,1)), (int(frame_size[1] * 0.45), int(frame_size[0] * 0.1)), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0),2)

		marker_num = len(markers)
		# if information_visual == True:
		# 	print(markers)
		# 	print(marker_num)
		if game_play == False:
			if marker_num == 2*2:
				if marker_set == 0:
					marker_set = time.time()
					start_count = 3
				else:
					if time.time() - marker_set > 1.5:
						game_play = True
						start_time = time.time()
						cv2.putText(frame, str("GAME START"), (int(frame_size[1] * 0.40), int(frame_size[0] * 0.5)),
												cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 2)
			else:
				if start_count > 0:
					start_count -= 1
				else:
					marker_set = 0
		else:
			for marker in markers:
				if marker.id == p1_id:
					p1_pose = marker.contours
					p1_center = marker.center
					p1_rotation = marker.rotation
				elif marker.id == p2_id:
					p2_pose = marker.contours
					p2_center = marker.center
					p2_rotation = marker.rotation
			if len(p1_pose) == 4:
				cv2.fillPoly(frame, [p1_pose], (255, 0, 0))
				# print(p1_center, p1_pose)
				cv2.line(frame, p1_center, p1_pose[p1_rotation][0], 5 )
			if len(p2_pose) == 4:
				cv2.fillPoly(frame, [p2_pose], (0, 0, 255))
				cv2.line(frame, p2_center, p2_pose[p2_rotation][0], 5)


		if information_visual == True:
			for marker in markers:
				marker.highlite_marker(frame)
				print(marker.id, 'contours', marker.contours)

		with mp_hands.Hands(
			model_complexity=0,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5) as hands:
			image = frame
			# To improve performance, optionally mark the image as not writeable to
			# pass by reference.
			image.flags.writeable = False
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			results = hands.process(image)

			# Draw the hand annotations on the image.
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			if results.multi_hand_landmarks:
				for hand_landmarks in results.multi_hand_landmarks:
					mp_drawing.draw_landmarks(
						image,
						hand_landmarks,
						mp_hands.HAND_CONNECTIONS,
						mp_drawing_styles.get_default_hand_landmarks_style(),
						mp_drawing_styles.get_default_hand_connections_style())
			frame = image

		cv2.imshow('Test Frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break



	# When everything done, release the capture
	capture.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main(sys.argv[1:])


