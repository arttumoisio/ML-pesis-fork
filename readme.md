# Näin käytät tätä härveliä:

- Lataa video:
  - http://admin-prod.kepit.tv/
  - Suosittelen 720P HQ
- Ota clipit losslesscutilla.
- Tallenna clipit kansioon
- Aja pitching_overlay.py -f "path to folder"
- Tarkastele tulosta kansiosta predictions
  - Ajon nimi on kansio_HH:MM_parametrit.mp4
- Tarvittaessa muokkaa parametrejä src/config.py ja toista ajo.

# Näin kehität mallia:

Lataa video:
http://admin-prod.kepit.tv/
720P HQ lienee hyvä laatu

Clippaa video:
ffmpeg -ss 00:30:30 -i XXXXXXXX.mp4 -c copy -to 01:31:20 1_jakso_clip.mp4
ffmpeg -ss 01:48:30 -i XXXXXXXX.mp4 -c copy -to 02:48:20 2_jakso_clip.mp4

videon croppaus 720P:

<!-- 1x2 -->

ffmpeg -i 1_jakso_clip.mp4 -filter:v "crop=480:720:400:0" 1_jakso_crop.mp4
ffmpeg -i 2_jakso_clip.mp4 -filter:v "crop=480:720:400:0" 2_jakso_crop.mp4

<!-- 1x1 -->

ffmpeg -i 1_jakso_clip.mp4 -filter:v "crop=720:720:280:0" 1_jakso_crop.mp4
ffmpeg -i 2_jakso_clip.mp4 -filter:v "crop=720:720:280:0" 2_jakso_crop.mp4

- Ota clipit losslesscutilla.
- Uploadaa clipit -> roboflow
- Annotointi
- uuden version luonti
- Opeta uudella versiolla -> train.py
- Ota uusi malli käyttöön lisäämällä runs/detect/trainX/weights/best.pt -> src/config.py[modelPath]
