from simple_websocket_server import WebSocketServer, WebSocket
import base64, cv2
import numpy as np
import warnings
import torch
from torchvision import transforms
from PIL import Image
from janus_client import JanusSession, JanusVideoCallPlugin
import io

import ffmpeg




warnings.simplefilter("ignore", DeprecationWarning)

model = torch.jit.load('./epoch150_netG.pt')
model = torch.nn.DataParallel(model)
transform = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
model.eval()

output_path = "output.mp4"  # 저장할 동영상 파일 경로 및 이름
frame_rate = 6  # 초당 프레임 수 (원하는 값으로 설정)
# VideoWriter_fourcc는 동영상 코덱을 설정합니다.
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 코덱 사용

# VideoWriter 객체 생성
v = cv2.VideoWriter(output_path, fourcc, frame_rate, (320, 320))

# janus_url = "http://localhost:8088/janus"
# room_id = 1234
# publisher_id = 333
# display_name = "qweqwe"
# # Janus 연결 설정
# # Janus 세션 설정
# session = JanusSession(janus_url)
# videoroom = session.attach("janus.plugin.videoroom")

class ImgRender(WebSocket):
    def handle(self):   
        ## msg = 클라이언트가 보낸 영상 
        msg = self.data
        ## 모델 적용 부분 
        image_data = base64.b64decode(msg.split(',')[1]) ## 디코딩
        image = Image.open(io.BytesIO(image_data))
        t_img=transform(image)
        t_img=t_img.view(1,3,320,320)
        with torch.no_grad():
            out=model(t_img)
        out=out.to('cpu')
        out=out*0.5+0.5       
        image_np = out[0].numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = (image_np * 255).astype(np.uint8)
        v.write(image_np)
        cv2.imshow("image", image_np)
        cv2.imwrite('a.png',image_np)

        cv2.waitKey(1)  # 키 입력 대기 시간을 1ms로 설정
        
    def connected(self):
        print(self.address, 'connected')

    def handle_close(self):
        print(self.address, 'closed')

server = WebSocketServer('localhost', 3000, ImgRender)
server.serve_forever()
