import argparse
import maskrcnn3d

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, help = "path to the input video file")
ap.add_argument("-o", "--output", required = True, help = "path to the output video file")
ap.add_argument("-w", "--width", help = "pixel width of the blank bar", default = 20)
ap.add_argument("-d", "--distance", type=int, choices=range(10,41), default = 15, help = "distance of the bar from the center, choice from int[10, 40]")
ap.add_argument("-c", "--color", choices=["white", "black"], help = "color of the blank bar, can be set to white or black", default = "white")

args = vars(ap.parse_args())
input_file = args["input"]
output_file = args["output"]
bar_width = int(args["width"])
bar_distance = int(args["distance"])
bar_color = args["color"]

test = maskrcnn3d.MRCNN3d(input_file, output_file, bar_width, bar_distance, bar_color)
test.convert()