from transformers import (
    DetrImageProcessor,
    TableTransformerForObjectDetection
)
from PIL import Image
import torch


class TableDetector:

    def __init__(self):
         #load image processor,it resize the image, normalize pixels
        self.processor = (DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection"))
        #load detection model it predicts where tables are ,bounding boxes
        self.model = (TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection"))

    def detect(self, image): #PDF page as NumPy array                                   

        pil_img = Image.fromarray(image)

        inputs = self.processor(images=pil_img,return_tensors="pt")
        # Sends image into AI model
        # Model predicts table locations
        outputs = self.model(**inputs)  
        #converts raw model output → usable bounding boxes
        results = (self.processor.post_process_object_detection(outputs,threshold=0.6,target_sizes=torch.tensor([pil_img.size[::-1]]))[0])

        tables = []

        for box in results["boxes"]:
            tables.append(box.tolist())

        return tables  #Return coordinates of tables  [50, 300, 400, 700]