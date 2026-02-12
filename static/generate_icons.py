from PIL import Image

src = 'static/images/alex-profile.webp'
out1 = 'static/images/icon-192.png'
out2 = 'static/images/icon-512.png'

img = Image.open(src).convert('RGBA')
img192 = img.resize((192,192), Image.LANCZOS)
img512 = img.resize((512,512), Image.LANCZOS)
img192.save(out1)
img512.save(out2)
print('Icons written:', out1, out2)
