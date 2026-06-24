from paddleocr import PaddleOCR


class OCREngine:

    def __init__(self):

        self.ocr = PaddleOCR(use_angle_cls=True,lang="en") #This loads the OCR engine.

    def extract(self, image):

        result = self.ocr.ocr(image) #Sends image to PaddleOCR,Gets detected text + coordinates

        words = []

        if not result or not result[0]:
            return words

        for line in result[0]:

            words.append({"bbox": line[0],"text": line[1][0]})

        return words  #returns "text","bbox(boundingbox-->Where exactly is this text located in the image)"