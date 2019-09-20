import numpy as np
import PIL.ImageDraw as draw
import PIL.Image as image
import PIL.ImageFont as Font

font = Font.truetype(font='1.ttf',size=36)
def num():
    return chr(np.random.randint(48,57))

def for_color():
    return (np.random.randint(100,200),
            np.random.randint(100,200),
            np.random.randint(100,200))

def back_color():
    return (np.random.randint(50,100),
            np.random.randint(50,100),
            np.random.randint(50,100))

h = 60
w = 120
for i in range(250):
    img=image.new('RGB',(w,h),color=(255,255,255))
    draw1=draw.ImageDraw(img)
    for m in range(w):
        for n in range(h):
            draw1.point((m,n),fill=back_color())
    fx = []
    for k in range(4):
        ch = num()
        fx.append(ch)
        draw1.text((k*30+10,10),text=ch,fill=for_color(),font=font)
    # print(fx)
    image_path = r'code/'
    img.save('{0}/{1}.jpg'.format(image_path,''.join(fx)))
    # img.show()


