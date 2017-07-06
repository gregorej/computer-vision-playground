import cv2


class FramesGenerator(object):

    def __init__(self, file_name):
        self.file_name = file_name
        self.cap = cv2.VideoCapture(file_name)

    def __iter__(self):
        return self

    def next(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            raise StopIteration()


def frames(video_file_name):
    return FramesGenerator(video_file_name)


class SlidingWindowsGenerator(object):

    def __init__(self, frame, window_width, window_height, step_x, step_y):
        self.frame = frame
        self.size = frame.shape[:2]
        self.ww = window_width
        self.wh = window_height
        self.sx = step_x
        self.sy = step_y
        self.x = 0
        self.y = 0

    def __iter__(self):
        return self

    def next(self):
        if self.x + self.ww > self.size[1]:
            self.x = 0
            self.y += self.sy
        if self.y + self.wh > self.size[0]:
            raise StopIteration()
        result = ((self.x, self.y), (self.x + self.ww, self.y + self.wh))
        self.x += self.sx
        return result

    def move_up_by(self, pixels):
        if self.y - pixels >= 0:
            self.y -= pixels

    def move_down_by(self, pixels):
        if self.y + pixels + self.wh <= self.size[0]:
            self.y += pixels

    def move_left_by(self, pixels):
        if self.x - pixels >= 0:
            self.x -= pixels

    def move_right_by(self, pixels):
        if self.x + pixels + self.ww <= self.size[1]:
            self.x += pixels

    def set_bbox(self, new_bbox):
        self.size = new_bbox

    def draw_on(self, image):
        with_sliding_window = image.copy()
        cv2.rectangle(with_sliding_window, (self.x, self.y), (self.x + self.ww, self.y + self.wh), (0, 255, 0), 2)
        return with_sliding_window


def sliding_windows(frame, width, height, step_x, step_y):
    return SlidingWindowsGenerator(frame, width, height, step_x, step_y)