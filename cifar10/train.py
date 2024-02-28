def training(epochs, total_steps, loader, Optimizer, Criterion, Model):
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(loader):
            #forward pass
            outputs = Model(imgs)
            loss = Criterion(outputs, labels)

            #backward pass and optimize
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()

            if (i+1) % 6250 ==0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')

    print("Finished Training")
    return Model