# Python-DiscordBot-FaceRecognition
Python Discord Bot Face detected and recognition, trained by local images

For any questions do not hesitate to write to my email: Santiagolautarolucero@gmail.com

Basically the process is it:

-Drop the image in the channel that the bot its allowed to read, add the command /who in the description of the message

-The script will get the http Url, and will pass that like an argument to a function (url_to_script)
-Using the haar cascade method, the script will detect the face coordinates on the img, and going to compare this information (like a nunpy array), with the features
of the trained face recognizer, and will asociate with his respective label.

-After that, will draw a rectangle over the face, put the label and return this to the Discord chat.
