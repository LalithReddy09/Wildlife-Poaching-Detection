import cv2
from ultralytics import YOLO

model = YOLO("/Users/divakar/Desktop/Yolo train/poaching/runs/detect/train/weights/best.pt")  # load a pretrained model (recommended for training)
cap = cv2.VideoCapture("/Users/divakar/Desktop/Yolo train/poaching/Output Video 2.mp4")
c=0
while True:
    ret, frame = cap.read()
    results = model(frame, device="mps", conf=0.1)

    class_names = ['-', '- annotate- and create datasets', '- collect - organize images', '- export- train- and deploy computer vision models', '- use active learning to improve your dataset over time', 'Roboflow is an end-to-end computer vision platform that helps you', 'This dataset was exported via roboflow.com on October 18- 2023 at 7-19 PM GMT', 'objectdetect - v2 2023-10-19 12-44am', 'poacher', 'weapon']

    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        probs = result.probs  # Class probabilities for classification outputs
        cls = boxes.cls.tolist()  # Convert tensor to list
        xyxy = boxes.xyxy
        conf = boxes.conf
        xywh = boxes.xywh  # box with xywh format, (N, 4)
        for class_index in cls:
            class_name = class_names[int(class_index)]
            print("Class: sdsaf", class_name)
    print(type(results))
    c+=1
    if c==1: break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# account_sid='AC714c701c7ee80855bc13e860bfaefbd3'

# auth_token = 'c3542470a4b76b2e5f7f5f3fcdc3896d'

# twilio_phone_number = 'whatsapp:++14157636913'

# recipient_phone_numbers = ['whatsapp:+918179533097']

# # Create a Twilio client
# client = Client(account_sid, auth_token)

# def send_whatsapp_message(message_content,to):
#     # Send the message
#     message = client.messages.create(
#         body=message_content,
#         from_=twilio_phone_number,
#         to=to
#     )

#     # Print the message SID (optional)
#     print(f"Message SID: {message.sid}")

# event_message_content = "There is an abnormal event occured"
# message_sent = False


# if 'weapon' == class_name and not message_sent:
#         print('Abnormal Event Detected')
#         for recipient in recipient_phone_numbers:
#             send_whatsapp_message(event_message_content, recipient)
#         message_sent = True