import torch


def fgsm_attack(model, image_tensor, ):
    epsilon=50
    device = next(model.parameters()).device  # get the model's device
    image_tensor = image_tensor.clone().detach().unsqueeze(0).to(device).requires_grad_(True)
    #print(epsilon)
    model.eval()
    recon = model(image_tensor)
    loss = torch.nn.functional.mse_loss(recon, image_tensor)
    loss.backward()

    adv_image = image_tensor + epsilon * image_tensor.grad.sign()
    adv_image = torch.clamp(adv_image, 0, 1)


    return adv_image.detach().cpu().squeeze(0), loss.item()



def pgd_attack(model, image_tensor, epsilon=0.1, alpha=0.01, steps=10):
    model.eval()
    original = image_tensor.clone().detach().unsqueeze(0)
    perturbed = original.clone().detach()

    for _ in range(steps):
        perturbed.requires_grad = True
        recon = model(perturbed)
        loss = torch.nn.functional.mse_loss(recon, original)
        loss.backward()

        # Gradient ascent on error
        perturbation = alpha * perturbed.grad.sign()
        perturbed = perturbed + perturbation
        perturbed = torch.clamp(perturbed, original - epsilon, original + epsilon)
        perturbed = torch.clamp(perturbed, 0, 1).detach()

    return perturbed.squeeze(0), loss.item()