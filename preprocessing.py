import sqlite3
import warnings
warnings.filterwarnings("ignore")
import cv2
from matplotlib import pyplot as plt
# Reading color image as grayscale
conn = sqlite3.connect('Form.db')
with conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM imgsave")
    rows = cursor.fetchall()
    for row in rows:
        filename = row[0]




gray = cv2.imread(filename,0)

# Showing grayscale image
cv2.imshow("Grayscale Image", gray)

# waiting for key event
cv2.waitKey(0)

# destroying all windows
cv2.destroyAllWindows()

img = cv2.imread(filename)

dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
cv2.imshow("Seeds", dst)

# waiting for key event
cv2.waitKey(0)
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()
