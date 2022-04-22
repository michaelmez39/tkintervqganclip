class Prompt:
    def init(prompt):
        self.embedded = CLIP.parse(prompt)
    def forward(input):
        return distance(input, self.embedded)

def synthesize_image(z):
    return VQGAN.decode(z)

def ascend_txt(optimizer, prompts):
    img = synthesize(optimizer.weight)
    losses = []
    for prompt in prompts:
        loss = prompt(img)
        losses.append(loss)

def train(optimizer, prompts):
    optimizer.zero_gradients()
    losses = sum(ascend_txt(z, prompts))

def generate_image(iterations, input_prompts):
    z = random_noise(size(latent_space))
    optimizer = Adam(z, learning_rate)
    prompts = [Prompt(text) for text in input_prompts]
    for iteration in range(iterations):
        image = train(optimizer, prompts)
        save(image, iteration)