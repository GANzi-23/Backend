from simple_websocket_server import WebSocketServer, WebSocket
import base64, cv2
import numpy as np
import warnings
import torch
from torchvision import transforms
from PIL import Image
import io

warnings.simplefilter("ignore", DeprecationWarning)

model = torch.jit.load('./epoch10_netG.pt')
model = torch.nn.DataParallel(model)
transform = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
model.eval()

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

        ## 실시간 출력을 위한 부분 numpy 변환 후 opencv 로         
        image_np = out[0].numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = (image_np * 255).astype(np.uint8)
        cv2.imshow("image", image_np)
        cv2.waitKey(1)  # 키 입력 대기 시간을 1ms로 설정
        
    def connected(self):
        print(self.address, 'connected')

    def handle_close(self):
        print(self.address, 'closed')

server = WebSocketServer('localhost', 3000, ImgRender)
server.serve_forever()
