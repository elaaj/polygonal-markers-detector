import cv2 as cv
import numpy as np
from math import cos, sin, radians


# ! Generates a list of point represeting a line between the two given points.
# ! The list will always start from (x0, y0), which in this use-case means from
# ! the concave corner of the related marker.
# ! It returnsa list of points representing a line between the two given
# ! points.
def bresenhamLineGenerator(x0, y0, x1, y1):
    """Generates a list of point represeting a line
    between the two given points. The list will always
    start from (x0, y0), which in this use-case means
    from the concave corner of the related marker.

    Args:
        x0 (int): point 0's x coordinate.
        y0 (int): point 0's y coordinate.
        x1 (int): point 1's x coordinate.
        y1 (int): point 1's y coordinate.

    Returns:
        list[int]: list of points.
    """

    # ! This calculates the absolute differences between the x and y coordinates
    # ! of the two points.
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    # This deals with division by zero.
    # The slope value needs just to be positive or negative,
    # hence any value can fit to handle this case (e.g., 10.).
    # ! This calculates the slope of the line between the two points.
    # ! If the `dx` value is 0 (meaning the line is vertical), the slope
    # ! is set to 10.0 (a large arbitrary value).
    slope = dy / dx if dx != 0 else 10.0

    # ! it is used to determine if the current iteration is the first
    # ! iteration of the loop. On the first iteration, the code inside the
    # ! `if` block is executed. This block initializes the variables `xPrevious`,
    # ! `pPrevious`, and `p`, and sets `flag` to `False`.
    flag = True

    # ! This initializes an empty list called `linePixel` and appends the
    # ! starting point to it.
    # ! It is used to contatin all points that make up the line between
    # ! the given points.
    linePixel = []
    linePixel.append((x0, y0))

    # Step initialized according to the type of slope
    # and position of X and Y.
    # ! These lines determine the step size of x and y when iterating
    # ! through the pixels that make up the line. The yStep is set to -1
    # ! if the slope is greater than or equal to 1.0 and y0 is greater
    # ! than y1. Otherwise, it is set to 1. The xStep is determined based
    # ! on several conditions that depend on the slope and the relative
    # ! positions of the two points. These conditions are used to determine
    # ! whether the line is going left or right.
    yStep = -1 if slope >= 1.0 and y0 > y1 else 1
    xStep = (
        -1
        if ((x0 > x1 or y0 > y1) and slope < 1.0)
        or (slope > 1.0 and x0 > x1 and y0 > y1)
        or (x0 > x1 and y0 < y1)
        else 1
    )

    # ! Check if the slope is smaller than one, and swap x and y if true.
    # ! "slopeSmallerThanOne = False" because it is assumed that the slope
    # ! is greater than or equal to one, unless proven otherwise.
    slopeSmallerThanOne = False
    # ! However, if the slope is found to be less than one (i.e., slope < 1),
    # ! the variables x0, x1, y0, y1 are swapped, and dx and dy are
    # ! recalculated accordingly. At this point, slopeSmallerThanOne is set
    # ! to True to indicate that the slope is less than one.
    if slope < 1:
        x0, x1, y0, y1 = y0, y1, x0, x1
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        slopeSmallerThanOne = True

    # ! Compute the first value of the Bresenham algorithm
    p0 = 2 * dx - dy  # ! is used to determine which pixel to draw next.
    # ! it is defined based on the difference between the
    # ! endpoints of the line segment.
    # ! are the initial pixel coordinates, and they are used to represent
    # ! the current point on the line as the algorithm iterates.
    x = x0
    y = y0

    # This method is invoked quite often, and each for loop
    # iterates an hundred times on average, so I chose to keep
    # it redundant with the two different appends, instead of
    # keeping a single for-loop with a control in it.
    # This loop proceeds step by step into generating the line.
    # Both X and Y require a dynamic step because of the different
    # slopes to handle.
    # If the slope is smaller than one, x and y will be appended
    # as (y, x), and not (x, y).
    if slopeSmallerThanOne:
        # ! This loop generates the line when the slope is smaller than one.
        # ! It loops for the difference between the y-coordinates of the two points.
        for _ in range(abs(y1 - y0)):
            # ! Flag is used to identify the first iteration of the loop.
            # ! This is needed to set the previous x and previous p values to the initial ones.
            # ! p stores the distance between the previous pixel and the ideal pixel on the line.
            # ! p = 2 * dx - dy
            # ! flag is then setted to false, because we are no longer in the first iteration.
            if flag:
                xPrevious = x0
                pPrevious = p0
                p = p0
                flag = False
            else:
                xPrevious = x
                pPrevious = p

            # ! If p is greater than or equal to zero, the next pixel to be added to the line is
            # ! in the next x position.
            if p >= 0:
                x = x + xStep

            # ! Update the p value for the next iteration.
            # ! When the slope is less than one, we add 2 * dx to p when the next x position is reached.
            # ! Otherwise, we subtract 2 * dy * (abs(x - xPrevious)) from p.
            p = pPrevious + 2 * dx - 2 * dy * (abs(x - xPrevious))
            y = y + yStep
            linePixel.append((y, x))
    else:
        for _ in range(abs(y1 - y0)):
            if flag:
                xPrevious = x0
                pPrevious = p0
                p = p0
                flag = False
            else:
                xPrevious = x
                pPrevious = p

            if p >= 0:
                x = x + xStep

            p = pPrevious + 2 * dx - 2 * dy * (abs(x - xPrevious))
            y = y + yStep
            linePixel.append((x, y))

    return linePixel


def computeDistance(p1, p2):
    """Compute the euclidean distance between p1 and p2.

    Args:
        p1 (tuple[int]): first point.
        p2 (tuple[int]): second point.

    Returns:
        float: Euclidean distance between p1 and p2.
    """
    # ! The formula for the Euclidean distance is the square root of the
    # ! sum of the squared differences between the x and y coordinates of
    # ! the two points.
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5


def detectAndLabelMarkers(
    image: np.ndarray, currentFrame: int, objectToTrack: int
) -> None:
    """Detect the visible markers through their contours and then determine for
    each of them the line which crosses all the circles from the bottom of the marker
    to the concave corner. Eventually this line is used to traverse the marker, looking
    for the white circles in fixed positions.

    Args:
        image (np.ndarray): input image.
        currentFrame (int): index of the current image with respect to the total number
        of frames in the video.
        objectToTrack (int): index of the chosen video. Required to write the rows in the
        related csv file.
    """

    # ! Initialize an empty string to hold the output file content
    outputFileContent = ""

    # ! Convert the input image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Discarded part of the image because the smallest
    # visible markers tend to be misdetected being them
    # adjacent to the plastic cup.
    gray[:, 0:1200:] = 0
    # 190 detects markers pretty well, but still requires an
    # area control for small fake-markers appearing on the plastic
    # cup.
    _, thresh = cv.threshold(gray, 190, 255, cv.THRESH_BINARY)

    # I use CHAIN_APPROX_SIMPLE because it removes all redundant points
    # and compresses the contour, thereby saving memory.
    # ! Find the contours (i.e., the boundaries) of the white objects in the
    # ! binary image using a hierarchical contour retrieval mode (RETR_TREE)
    # ! and compress the contours by removing redundant points to save
    # ! memory.
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Searching through every region selected to find the required polygon.
    # ! Find the polygons with 5 vertices (pentagons) among the contours
    # ! with area greater than 1200. The threshold of 1200 is used to
    # ! filter out small fake-markers appearing on the plastic cup.
    polygons = [
        approx
        for cnt in contours
        if cv.contourArea(cnt) > 1200
        and len(approx := cv.approxPolyDP(cnt, 0.0155 * cv.arcLength(cnt, True), True))
        == 5
    ]
    # ! Draw the polygons found on the original image with a green line
    # ! of width 2 pixels.
    cv.drawContours(image, polygons, -1, (0, 255, 0), 2)

    # Cycle through every marker, and find for each of them the point A.
    # Take the 3 shortest sides of each marker: the A point will be
    # the one where two of such sides meet.
    for poly in polygons:
        # ! Initialize variables to hold the concave corner point and the
        # ! middle point on the lower side of the marker.
        concaveCornerPoint = None
        lowerSideMiddlePoint = None
        # ! Count the number of vertices in the polygon
        totalVertices = len(poly)
        # Retreving the 3 shortest sides for the actual polygon.
        # I chose 80 as filter because sides larger then 80 are
        # the long sides, which are not useful for the detection
        # of A.
        # ! For each vertex it checks if the distance between the vertex and the next vertex in the
        # ! polygon is less than 80. If the distance is less than 80, the vertexIndex is added to the list.
        matchingSides = [
            vertexIndex
            for vertexIndex, vertex in enumerate(poly)
            if computeDistance(vertex[0], poly[(vertexIndex + 1) % totalVertices][0])
            < 80.0
        ]
        # Iterate through the 3 sides and control them by couples to
        # find which of them match
        # ! Loop through each side in matchingSides. Set initial values for the four points
        # ! that define the two sides.
        for loopIndex, side in enumerate(matchingSides):
            cornerFound = False

            # Legend:
            #   - Side: index of the first vertex of the currently examined side.
            #   - (side + 1) % totalVertices: index of the second vertex of the currently examined side.
            #   - matchingSides[(loopIndex + 1) % 3]: index of the first vertex of the next matching side.
            #   - (matchingSides[(loopIndex + 1) % 3] + 1) % totalVertices: index of the second vertex
            #                                                               of the next matching side.
            firstSideFirstPoint = poly[side][0]
            firstSideSecondPoint = poly[(side + 1) % totalVertices][0]
            secondSideFirstPoint = poly[matchingSides[(loopIndex + 1) % 3]][0]
            secondSideSecondPoint = poly[
                (matchingSides[(loopIndex + 1) % 3] + 1) % totalVertices
            ][0]

            # Check if the two currently examined sides have a common point: if so,
            # store it for the next operations, and record that the matching lines
            # have been found for this marker. If no match is found, keep iterating.
            # ! Check if the two sides intersect at one of their endpoints. If so, set cornerFound
            # ! to True and store the corner point in concaveCornerPoint.
            # ! FIXME: check if the defintion is correct
            # ! I have used the function "all" that checks that all the elements of the array are equal.
            if (firstSideFirstPoint == secondSideFirstPoint).all() or (
                firstSideFirstPoint == secondSideSecondPoint
            ).all():
                cornerFound = True
                concaveCornerPoint = firstSideFirstPoint
            elif (firstSideSecondPoint == secondSideFirstPoint).all() or (
                firstSideSecondPoint == secondSideSecondPoint
            ).all():
                cornerFound = True
                concaveCornerPoint = firstSideSecondPoint

            # ! If a corner was found, draw a red circle at the corner point.
            if cornerFound:
                cv.circle(
                    image,
                    (
                        concaveCornerPoint[0],
                        concaveCornerPoint[1],
                    ),
                    radius=1,
                    color=(0, 0, 255),
                    thickness=6,
                )

                # "A" was found, so I use it to generate a line between it and
                # the middle point on the lower side of the marker: I use this
                # line to look in the 5 areas where each circle should reside,
                # obtaining in such way the binary value of each marker.
                # ! Calculate the midpoint of the side opposite the concave corner
                # ! and store it in lowerSideMiddlePoint.
                lowerSide = (
                    poly[matchingSides[(loopIndex + 2) % 3]][0],
                    poly[(matchingSides[(loopIndex + 2) % 3] + 1) % totalVertices][0],
                )
                middlePointX = (lowerSide[0][0] + lowerSide[1][0]) // 2
                middlePointY = (lowerSide[0][1] + lowerSide[1][1]) // 2
                lowerSideMiddlePoint = (middlePointX, middlePointY)

                # Use the found middle point and "A" to generate the line
                # between them.
                markerAxis = bresenhamLineGenerator(
                    concaveCornerPoint[0],
                    concaveCornerPoint[1],
                    lowerSideMiddlePoint[0],
                    lowerSideMiddlePoint[1],
                )
                # ! FIXME: DA SPIEGARE!!
                markerAxisLen = len(markerAxis)
                actCycle = 0
                cycleJump = int(markerAxisLen / 10 * 1.95)
                binaryRepr = ""
                for intervalCentreIndex in range(cycleJump, markerAxisLen, cycleJump):
                    tempIndex = intervalCentreIndex

                    # This if-elif block performs a little correction on the
                    # place where to look for the white circles: theoretically,
                    # the line generated through bresenham should be splitted into
                    # 5 pieces, and the detection of each circle should be made by
                    # looking into the centers of such pieces, but the line is
                    # perspectively warped, so an adjustment is required when
                    # looking for each circle's center.
                    if actCycle in [1, 2, 3, 4]:
                        tempIndex = int(tempIndex * 0.85)
                    elif actCycle == 0:
                        tempIndex = int(tempIndex * 0.9)
                    actCycle += 1

                    # Store the current slot status: white or black.
                    # (180 seems a good threshold to discriminate them
                    # according to some prints).
                    binaryRepr += (
                        "0"
                        if gray[
                            markerAxis[tempIndex][1],
                            markerAxis[tempIndex][0],
                        ]
                        > 180
                        else "1"
                    )
                    # Draw the center of the actual circle.
                    cv.circle(
                        image,
                        (
                            markerAxis[tempIndex][0],
                            markerAxis[tempIndex][1],
                        ),
                        radius=1,
                        color=(255, 0, 0),
                        thickness=2,
                    )
                # Reverse the binary representation, which was captured backward,
                # and write it on the marker in decimal representation.
                # Also, using the computed label, access the related 3D coords.
                binaryRepr = int(binaryRepr[::-1], 2)
                binaryReprStr = str(binaryRepr)

                # ! radians convert angle x from degrees to radians.
                # ! FIXME: Why is it "-15"?
                radAngle = radians(-15)
                qx = cos(radAngle * binaryRepr) * 70
                qy = sin(radAngle * binaryRepr) * 70
                outputFileContent = (
                    outputFileContent
                    + f"{currentFrame},{binaryReprStr},{markerAxis[0][0]},{markerAxis[0][1]},{qx},{qy},0\n"
                )  # ! The values includes the current frame number, a binary representation, the x and y coordinates of a marker axis,
                # ! and the calculated values of qx, qy, and 0 are then concatenated and stored in outputFileContent.
                image = cv.putText(
                    img=image,
                    text=binaryReprStr,
                    org=(markerAxis[0][0], markerAxis[0][1]),
                    fontScale=1.0,
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    color=(0, 0, 0),
                    thickness=9,
                )
                image = cv.putText(
                    img=image,
                    text=binaryReprStr,
                    org=(markerAxis[0][0], markerAxis[0][1]),
                    fontScale=1.0,
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    color=(255, 255, 255),
                    thickness=3,
                )
                break

    # ! It is opening a file named "obj{objectToTrack}_marker.csv" in append mode ("a")
    outputFile = open(f"obj{objectToTrack}_marker.csv", "a")
    # ! Then I write the content of the variable outputFileContent using write().
    outputFile.write(outputFileContent)
    outputFile.close()
