# Rota Image to GCal
## An image processing script that extracts shifts from a picture of a work schedule and adds them to GCal

1. It takes the picture and first detects my name
2. It then detects the days that I'm ON and OFF
3. It then creates a small picture out of each day, enlarges it, masks it (turning it into 255 white and 0 black) and dilates it
4. Finally it extracts the times, and the passes them to a function written in a separate file, that adds them to my GCal
