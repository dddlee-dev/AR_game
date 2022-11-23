import cv2
import mediapipe as mp
import sys, getopt
import time
import numpy as np
import random
import math

try:
	import cv2
	from ar_markers import detect_markers
except ImportError:
	raise Exception('Error: OpenCv is not installed')

# information_visual = True
information_visual = False
cam = 1 # default = 0

class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
#             elif values[0] in ('usemtl', 'usemat'):
#                 material = values[1]
#             elif values[0] == 'mtllib':
#                 self.mtl = MTL(filename.replace(".obj",".mtl"))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                #self.faces.append((face, norms, texcoords, material))
                self.faces.append((face, norms, texcoords))
def renderObj(img, obj, projection, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = (644,372)
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, color)
        else:
#             color = hex_to_rgb(face[-1])
            color = face[-1]
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img
def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

class bullet():
	def __init__(self, pos, vec, size):
		self.pos = pos + (vec * 1.05)
		self.life_time = 5
		self.t = time.time()
		self.vec = vec
		self.vec_d = (vec[0]**2+vec[1]**2)**0.5
		self.speed = 5
		self.size = size

	def attack(self, position):
		 # 포지션안에 불릿 pos위치시 작동
		self.pos
		p = 1
		return p

	def move(self):
		t1 = time.time()
		if t1 - self.t > self.life_time:
			del self
			return 3
		else:
			# print(self.pos, self.vec)
			self.pos[0] += (self.vec[0])*(t1 - self.t)/self.vec_d * self.speed
			self.pos[1] += (self.vec[1]) * (t1 - self.t)/self.vec_d * self.speed
			if self.pos[0] < 0 or self.pos[1] > self.size[0] or self.pos[1] < 0 or self.pos[1] > self.size[1]:
				del self
				return 3
		return 1

	def	__del__(self):
		return 3

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
	p1_b = []
	p2_b = []
	p1_ani = -1
	p2_ani = -1
	# ani = ['boy.obj', 'cow.obj', 'fox.obj', 'rat.obj', 'wolf.obj', 'pirate-ship-fat.obj']
	ani = ['boy.obj', 'fox.obj', 'rat.obj', 'wolf.obj', 'pirate-ship-fat.obj']
	obj1 = []
	obj2 = []

	p1_id = 2495
	p2_id = 482

	play_time = 120.0
	game_play = False
	start_time = 0
	marker_num = 0
	marker_set = 0
	start_count = 5
	info = 0
	info_time = 0

	hit_box = np.zeros((1080, 1920))
	camera_parameters = np.array([[800, 0, 320],
																[0, 800, 240],
																[0, 0, 1]])
	src_val = 800
	src_pts = np.float32([0, 0,
												src_val, 0,
												src_val, src_val,
												0, src_val]).reshape(-1, 1, 2)

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

		# print(frame.shape) #(720, 1280,3) (1080, 1920, 3)

		image = frame.copy()
		frame_size = frame.shape
		marker_num = len(markers)
		# if information_visual == True:
		# 	print(markers)
		# 	print(marker_num)
		if game_play == False:
			if marker_num == 2*2:
				if marker_set == 0:
					marker_set = time.time()
					start_count = 5
				else:
					if time.time() - marker_set > 1.5: #1.5:
						game_play = True
						start_time = time.time()
						info = 1
						p1_ani = random.randint(0,len(ani))
						p2_ani = random.randint(0, len(ani))
						obj1 = OBJ('./'+ani[p1_ani], swapyz=True)
						obj2 = OBJ('./'+ani[p2_ani], swapyz=True)

			else:
				if start_count > 0:
					start_count -= 1
				else:
					marker_set = 0
		else:
			cv2.rectangle(frame, (0,0),(int(frame_size[1]), int(frame_size[0])),
										(0, 0, 255), 10)
			for marker in markers:
				if marker.id == p1_id:
					p1_pose = marker.contours
					p1_center = marker.center
					p1_rotation = marker.rotation
				elif marker.id == p2_id:
					p2_pose = marker.contours
					p2_center = marker.center
					p2_rotation = marker.rotation
			hit_box = np.zeros(frame_size[:-1])
			# print(hit_box.shape)
			if len(p1_pose) == 4:
				cv2.fillPoly(frame, [p1_pose], (255, 0, 0))
				cv2.fillPoly(hit_box, [p1_pose], 1)
				# print(p1_center, p1_pose)
				cv2.line(frame, p1_center, p1_pose[p1_rotation][0], 5 )



				dst_pts = np.float32([p1_pose]).reshape(-1, 1, 2)
				dst_pts = dst_pts.round(2)
				# print('5', src_pts, dst_pts)
				homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
				projection = projection_matrix(camera_parameters, homography)
				frame = renderObj(frame, obj1, projection, (230, 20, 20))

			if len(p2_pose) == 4:
				cv2.fillPoly(frame, [p2_pose], (0, 0, 255))
				cv2.fillPoly(hit_box, [p2_pose], 2)
				cv2.line(frame, p2_center, p2_pose[p2_rotation][0], 5)


				dst_pts = np.float32([p2_pose]).reshape(-1, 1, 2)
				dst_pts = dst_pts.round(2)
				# print('5', src_pts, dst_pts)
				homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
				projection = projection_matrix(camera_parameters, homography)
				frame = renderObj(frame, obj2, projection, (20, 20, 230))

			p1_r = random.randint(1,10)
			p2_r = random.randint(1,10)
			if p1_r % 10 == 3:
				p1_b.append(bullet([p1_center[0],p1_center[1]], p1_pose[p1_rotation][0] - p1_center, frame_size[:-1]))
			if p2_r % 10 == 3:
				p2_b.append(bullet([p2_center[0],p2_center[1]], p2_pose[p2_rotation][0] - p2_center, frame_size[:-1]))



			for i in p1_b:
				d = i.move()
				if d == 3:
					p1_b.remove(i)
				else:
					if round(i.pos[1]) > 0 and round(i.pos[1]) < frame_size[1] and round(i.pos[0]) > 0 and round(i.pos[0]) < frame_size[0]:
						if hit_box[round(i.pos[1]), round(i.pos[0])] == 2:
							cv2.fillPoly(frame, [p2_pose], (60, 60, 200))
							cv2.line(frame, (round(i.pos[0]), round(i.pos[1])), (round(i.pos[0]), round(i.pos[1])), (255, 255, 255), 15)
							p2_hp -= 10
							p1_b.remove(i)
						else:
							cv2.line(frame, (round(i.pos[0]), round(i.pos[1])),(round(i.pos[0]), round(i.pos[1])), (200,0,125),10)
					# else:q
				# print((round(i.pos[0]), round(i.pos[1])))
			for i in p2_b:
				d = i.move()
				if d == 3:
					p2_b.remove(i)
				else:
					if round(i.pos[1]) > 0 and round(i.pos[1]) < frame_size[1] and round(i.pos[0]) > 0 and round(i.pos[0]) <frame_size[0]:
						if hit_box[round(i.pos[1]), round(i.pos[0])] == 1:
							cv2.fillPoly(frame, [p1_pose], (200, 60, 60))
							cv2.line(frame, (round(i.pos[0]), round(i.pos[1])), (round(i.pos[0]), round(i.pos[1])), (255, 255, 255), 15)
							p1_hp -= 10
							p2_b.remove(i)
						else:
							cv2.line(frame, (round(i.pos[0]), round(i.pos[1])), (round(i.pos[0]), round(i.pos[1])), (125, 0, 200), 10)
					# else:
					# 	p1_b.remove(i)
			if p1_hp <= 0:
				p1_hp = 0
				info = 2
				game_play = False
			if p2_hp <= 0:
				p2_hp = 0
				info = 3
				game_play = False

		cv2.rectangle(frame, (int(frame_size[1] * 0.04), int(frame_size[0] * 0.05)),
									(int(frame_size[1] * 0.44), int(frame_size[0] * 0.10)),
									(125,125,125), -1)
		cv2.rectangle(frame, (int(frame_size[1] * 0.04), int(frame_size[0] * 0.05)), ((int(frame_size[1] * 0.40 * (p1_hp / 100) + frame_size[1] * 0.04)), int(frame_size[0] * 0.10)),
									(255, 0, 0), -1)
		cv2.rectangle(frame, (int(frame_size[1] * 0.04), int(frame_size[0] * 0.05)),
									(int(frame_size[1] * 0.44), int(frame_size[0] * 0.10)),
									(0,0,0), 3)
		cv2.rectangle(frame, (int(frame_size[1] * 0.56), int(frame_size[0] * 0.05)),
									(int(frame_size[1] * 0.96), int(frame_size[0] * 0.10)),
									(125, 125, 125), -1)
		cv2.rectangle(frame, (int(frame_size[1] * 0.56), int(frame_size[0] * 0.05)), ((int(frame_size[1] * 0.40 * (p2_hp / 100) + frame_size[1] * 0.56)), int(frame_size[0] * 0.10)),
									(0, 0, 255), -1)
		cv2.rectangle(frame, (int(frame_size[1] * 0.56), int(frame_size[0] * 0.05)), (int(frame_size[1] * 0.96), int(frame_size[0] * 0.10)),
									(0,0,0), 3)



		if game_play == True:
			play_time = 120 - (time.time() - start_time)
			if play_time < 0:
				play_time = 0
				game_play = False
		# cv2.rectangle(frame, (int(frame_size[1] * 0.45), int(frame_size[0] * 0.05)),
		# 							(int(frame_size[1] * 0.55), int(frame_size[0] * 0.10)),
		# 							(255, 255, 255), -1)
		cv2.putText(frame, str(round(play_time,1)), (int(frame_size[1] * 0.45), int(frame_size[0] * 0.1)), cv2.FONT_HERSHEY_DUPLEX, 1.5, (50, 50, 50),2)

		if info == 1:
			if info_time == 0:
				info_time = time.time()
			else:
				if time.time() - info_time > 3:
					info = 0
					info_time = 0
			cv2.putText(frame, str("GAME START"), (int(frame_size[1] * 0.40), int(frame_size[0] * 0.5)),
									cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 2)
		elif info == 2:
			if info_time == 0:
				info_time = time.time()
			else:
				if time.time() - info_time > 3:
					info = 0
					info_time = 0
					p1_hp = 100
					p2_hp = 100
					p1_ani = -1
					p2_ani = -1
			marker_set = 0
			cv2.rectangle(frame, (int(frame_size[1] * 0.3), int(frame_size[0] * 0.3)),
										((int(frame_size[1] * 0.7)), int(frame_size[0] * 0.7)),
										(0, 0, 200), -1)
			cv2.putText(frame, 'red Win !!', (int(frame_size[1] * 0.3), int(frame_size[0] * 0.5)),
									cv2.FONT_HERSHEY_DUPLEX, 5, (255, 255, 255), 2)
		elif info == 3:
			if info_time == 0:
				info_time = time.time()
			else:
				if time.time() - info_time > 3:
					info = 0
					info_time = 0
					p1_hp = 100
					p2_hp = 100
					p1_ani = -1
					p2_ani = -1
			marker_set = 0
			cv2.rectangle(frame, (int(frame_size[1] * 0.3), int(frame_size[0] * 0.3)),
										((int(frame_size[1] * 0.7)), int(frame_size[0] * 0.7)),
										(200, 0, 00), -1)
			cv2.putText(frame, 'blue Win !!', (int(frame_size[1] * 0.3), int(frame_size[0] * 0.5)),
									cv2.FONT_HERSHEY_DUPLEX, 5, (255, 255, 255), 2)


		# if information_visual == True:
		for marker in markers:
			marker.highlite_marker(frame)
			print(marker.id, 'contours', marker.contours)

		with mp_hands.Hands(
			model_complexity=0,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5) as hands:
			# image = frame
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
						# image,
						frame,
						hand_landmarks,
						mp_hands.HAND_CONNECTIONS,
						mp_drawing_styles.get_default_hand_landmarks_style(),
						mp_drawing_styles.get_default_hand_connections_style())

					if hit_box[round(hand_landmarks.landmark[12].y*frame_size[0])][round(hand_landmarks.landmark[8].x*frame_size[1])] == 1:
						p1_b.append(bullet([p1_center[0], p1_center[1]], p1_pose[p1_rotation][0] - p1_center, frame_size[:-1]))
					elif hit_box[round(hand_landmarks.landmark[12].y*frame_size[0])][round(hand_landmarks.landmark[8].x*frame_size[1])] == 2:
						p2_b.append(bullet([p2_center[0], p2_center[1]], p2_pose[p2_rotation][0] - p2_center, frame_size[:-1]))

			# frame = image

		if information_visual == True:
			cv2.imshow('Test Frame2', hit_box)
		cv2.imshow('Test Frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break



	# When everything done, release the capture
	capture.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main(sys.argv[1:])


