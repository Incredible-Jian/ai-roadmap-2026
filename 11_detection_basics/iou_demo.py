def box_area(box):
	x1, y1, x2, y2 = box

	width = max(0.0, x2 - x1)
	height = max(0.0, y2 - y1)

	return width * height

def calculate_iou(box_a, box_b):
	intersection_x1 = max(box_a[0], box_b[0])
	intersection_y1 = max(box_a[1], box_b[1])

	intersection_x2 = min(box_a[2], box_b[2])
	intersection_y2 = min(box_a[3], box_b[3])

	intersection_width = max(
		0.0,
		intersection_x2 - intersection_x1
	)

	intersection_height = max(
		0.0,
		intersection_y2 - intersection_y1
	)

	intersection_area = (
		intersection_width * intersection_height
	)


	union_area = (
		box_area(box_a)
		+ box_area(box_b)
		- intersection_area
	)

	if union_area <= 0.0:
		return 0.0

	return intersection_area / union_area

ground_truth = [0.0, 0.0, 4.0, 2.0]

candidate_boxes = {
	"A_small" : [1.0, 0.5, 3.0, 1.5],
	"B_exact": [0.0, 0.0, 4.0, 2.0],
	"C_large": [-1.0, -1.0, 5.0, 3.0],
	"D_shifted": [3.0, 0.0, 7.0, 2.0],
	"E_separate": [5.0, 0.0, 7.0, 2.0],
	"F_my_box": [0.0, 0.0, 2.0, 2.0],
}

print("Ground-truth area:", box_area(ground_truth))

for name, candidate in candidate_boxes.items():
	area = box_area(candidate)
	iou = calculate_iou(candidate, ground_truth)

	print(
		f"{name:12s} | "
		f"Area: {area:.2f} | "
		f"IoU: {iou:.4f}"
	)
