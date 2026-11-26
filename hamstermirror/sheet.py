"""Contact sheet of numbered memes, used by the annotation tool."""
import cv2
import numpy as np


def contact_sheet(memes, cell=140, cols=8):
    rows = (len(memes) + cols - 1) // cols
    sheet = np.full((rows * (cell + 20), cols * cell, 3), 255, np.uint8)
    for i, m in enumerate(memes):
        im = cv2.cvtColor(np.array(m.load()), cv2.COLOR_RGB2BGR)
        s = cell / max(im.shape[:2])
        im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
        r, c = divmod(i, cols)
        y0, x0 = r * (cell + 20), c * cell
        sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
        cv2.putText(sheet, f"{i}: {m.name[:16]}", (x0 + 2, y0 + cell + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1, cv2.LINE_AA)
    return sheet
