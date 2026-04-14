print("🚀 OCR RUNNING")

import cv2
from Model import Model, DecoderType
from SamplePreprocessor import preprocessor
from DataLoader import Batch


# 🔁 CHANGE IMAGE NAME HERE
IMAGE_PATH = '../data/self.png'  # ✅ this image works


def run_ocr():
    print("Loading image...")

    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("❌ Error: Image not found")
        return

    # preprocess image
    img = preprocessor(img, Model.imgSize)

    print("Loading model...")

    model = Model(
        open('../data/charList.txt').read(),
        decoderType=DecoderType.BestPath,
        mustRestore=True   # ✅ loads trained model
    )

    print("Running OCR...")

    batch = Batch(None, [img])
    recognized = model.inferBatch(batch)

    print("\n✅ FINAL OUTPUT:")
    print("Recognized Text:", recognized[0])


# run function
if __name__ == "__main__":
    run_ocr()