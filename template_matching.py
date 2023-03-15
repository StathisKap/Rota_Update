# type: ignore
import sys
import argparse
import datetime
import cv2
import numpy as np
import pytesseract
from imutils.object_detection import non_max_suppression
from google_calendar import Create_Event

def Get_First_Day_Of_Week():
    now = datetime.datetime.utcnow()

    # Calculate the number of days to subtract from now to get the first day of the week (Monday)
    days_to_subtract = now.weekday()

    # Calculate the first day of the week
    first_day_of_week = now - datetime.timedelta(days=days_to_subtract)
    return datetime.datetime.combine(first_day_of_week, datetime.time.min)

def Str_To_Time(s):
    hours = int(s[:2])
    minutes = int(s[2:])
    time = [hours, minutes]
    time = datetime.time(hour=time[0], minute=time[1])
    return time

def Add_To_GCalendar(shifts):
    first_day_of_week = Get_First_Day_Of_Week()

    for i, shift in enumerate(shifts):
        if shift[0] == "OFF":
            continue
        if "," in shift[1]:
            days_shifts = shift[1].split(",")
        else:
            days_shifts = [shift[1]]

        for days_shift in days_shifts:
            days_shift_times = days_shift.split(" - ")
            days_shift_time_start = Str_To_Time(days_shift_times[0])
            days_shift_time_end = Str_To_Time(days_shift_times[1])

            start_time = first_day_of_week + datetime.timedelta(
                days=i,
                hours=days_shift_time_start.hour,
                minutes=days_shift_time_start.minute)

            end_time = first_day_of_week + datetime.timedelta(
                days=i,
                hours=days_shift_time_end.hour,
                minutes=days_shift_time_end.minute)

            #print(start_time, end_time)
            Create_Event(start_time, end_time)

def Text_Detection(img):
    ''' Takes in an image, processes it, and returns the text within it
    '''
    # Gray it out
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Create a binary mask using thresholding

    # Upscale the image by a factor of 2
    upscaled_img = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    thresh = cv2.threshold(upscaled_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Perform dilation on the upscaled image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    text = pytesseract.image_to_string(dilated)

    return text, dilated


def Match(img, temp_name, method, threshold, color):
    ''' Takes in an image and a template, a method to do the mathcing, and a threshold
    and draws a rectangle around it based on the color provided
    '''
    # Reading the image and the template
    if type(img) == str:
        img = cv2.imread(img)
    temp = cv2.imread(temp_name)

    # save the image dimensions
    H, W = temp.shape[:2]

    # Converting them to grayscale
    img_gray = cv2.cvtColor(img,
                            cv2.COLOR_BGR2GRAY)
    temp_gray = cv2.cvtColor(temp,
                             cv2.COLOR_BGR2GRAY)

    # Passing the image to matchTemplate method
    match = cv2.matchTemplate(
        image=img_gray, templ=temp_gray,
      method=method)

    # Select rectangles with
    # confidence greater than threshold
    (y_points, x_points) = np.where(match >= threshold)

    # initialize our list of rectangles
    boxes = list()

    # loop over the starting (x, y)-coordinates again
    for (x, y) in zip(x_points, y_points):

        # update our list of rectangles
        boxes.append((x, y, x + W, y + H))

    # apply non-maxima suppression to the rectangles
    # this will create a single bounding box
    boxes = non_max_suppression(np.array(boxes))

    # loop over the final bounding boxes
    for (x1, y1, x2, y2) in boxes:

        # draw the bounding box on the image
        cv2.rectangle(img, (x1, y1), (x2, y2),
                      color, 1)

    # crop the image to the range of detected templates
    lowest_y = min(y_points)
    img_cropped = img[lowest_y:lowest_y + H, :]
    return img, img_cropped, boxes

def Combine_Lists(list1, list2):

    # If it isn't 7, then it hasn't detected all the days, and thus we don't want to proceed
    if len(list1) + len(list2) != 7:
        sys.exit("Missed something")

    # Adds a label to the lists
    list1 = [sublist + ["ON"] for sublist in list1]
    list2 = [sublist + ["OFF"] for sublist in list2]


    # Puts them in order so that we know which day is which
    both_lists = list1 + list2
    sorted_list = sorted(both_lists, key=lambda x: x[0])
    return sorted_list

def main():


    # Create the argument parser
    parser = argparse.ArgumentParser(description='Provide a picture of the Rota which then extracts the shifts for Efstathios')

    # Add the argument for file_path
    parser.add_argument('--file_path','-f', type=str, help='the path to the image to be processed', default="Foh.jpeg")

    # Parse the arguments
    args = parser.parse_args()

    # Access the file_path argument value
    img_name = args.file_path


    assets_folder = "./assets/"
    Efstathios_template = assets_folder + "Efstathios-Kapnidis.jpg"
    ON_template = assets_folder + "ON.jpg"
    OFF_template = assets_folder + "OFF.jpeg"
    method = cv2.TM_CCOEFF_NORMED
    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)
    buffer = 15


    # Finds all the days on and off for Efstathios
    img, cropped_image, _ = Match(img_name, Efstathios_template, method, 0.9, red_color)
    img, _, ON_boxes = Match(cropped_image, ON_template, method, 0.6, blue_color)
    img, _, OFF_boxes  = Match(img, OFF_template, method, 0.52, blue_color)

    # Turns the numpy arrays to lists
    sorted_boxes = Combine_Lists(ON_boxes.tolist(), OFF_boxes.tolist())

    shifts = list()
    for _, box in enumerate(sorted_boxes):
        box_img = cropped_image[:,box[0] - buffer :box[2] + buffer]
#        cv2.imshow(f"box{i}", box_img)
#        text, txt_img = Text_Detection(box_img)
        text, _ = Text_Detection(box_img)
#        cv2.imshow(f"box_t{i}", txt_img)
        shifts.append(text)


    shifts_cleaned = list()
    for i, shift in enumerate(shifts):
        shift = shift.rstrip("\n").replace("\n",",")
        shifts_cleaned.append([sorted_boxes[i][4], shift])

    print(shifts_cleaned)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return shifts_cleaned

if __name__ == '__main__':
    shifts = main()
    Add_To_GCalendar(shifts)
