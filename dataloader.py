#([4=batch, 28=w, 28=s, 4=ch])
 
transformed_dataset = YourDataset(txt_path='data/labels.csv', img_dir='data/',
                  transform=transforms.Compose([
                      transforms.Resize(28),
                      transforms.CenterCrop(28),
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))
                  ]))

dataloader = DataLoader(transformed_dataset, batch_size=400,
                        shuffle=True, num_workers=0)
#https://stackoverflow.com/questions/384759/how-to-convert-a-pil-image-into-a-numpy-array
import matplotlib.pyplot as plt

s
for i_batch, sample_batched in enumerate(dataloader):
    print(torch.squeeze(sample_batched['image'][:1,:,:,:1]).shape)
    fig, subplots = plt.subplots(1, 4)
    fig.set_size_inches(15, 15)

    for i, s in enumerate(subplots.flatten()):
      image = torch.squeeze(sample_batched['image'][i,:,:,:1])
      s.imshow(np.reshape(image, [28, 28]), cmap='gray')
      s.axis('off')
    
