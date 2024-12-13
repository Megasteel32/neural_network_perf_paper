import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageFilter, ImageTk, ImageEnhance
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt


class EMNISTAugmenter(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

        # Pre-calculate augmentation parameters
        self.rotation_angles = torch.FloatTensor(len(images)).uniform_(-15, 15)
        self.shear_x = torch.FloatTensor(len(images)).uniform_(-0.1, 0.1)
        self.shear_y = torch.FloatTensor(len(images)).uniform_(-0.1, 0.1)
        self.translate_x = torch.FloatTensor(len(images)).uniform_(-3, 3)
        self.translate_y = torch.FloatTensor(len(images)).uniform_(-3, 3)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Ensure 4D format [B, C, H, W]
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif len(image.shape) == 3:
            image = image.unsqueeze(0)

        # Apply rotation
        angle = self.rotation_angles[idx]
        image = TF.rotate(image, angle.item(), fill=0)

        # Apply shear
        shear_matrix = torch.tensor([
            [1.0, self.shear_x[idx], 0.0],
            [self.shear_y[idx], 1.0, 0.0]
        ])

        grid = F.affine_grid(
            shear_matrix.unsqueeze(0),
            image.size(),
            align_corners=False
        )
        image = F.grid_sample(
            image,
            grid,
            align_corners=False,
            mode='bilinear',
            padding_mode='zeros'
        )

        # Apply translation
        translate_matrix = torch.tensor([
            [1.0, 0.0, self.translate_x[idx] / image.size(-1)],
            [0.0, 1.0, self.translate_y[idx] / image.size(-2)]
        ])

        grid = F.affine_grid(
            translate_matrix.unsqueeze(0),
            image.size(),
            align_corners=False
        )
        image = F.grid_sample(
            image,
            grid,
            align_corners=False,
            mode='bilinear',
            padding_mode='zeros'
        )

        return image.squeeze(0), label

def display_examples(images, labels, title, samples_per_class=10):
    """Display example digits from the dataset"""
    plt.figure(figsize=(15, 8))
    plt.suptitle(title)

    for digit in range(10):
        digit_indices = (labels == digit).nonzero().squeeze()

        if len(digit_indices) == 0:
            continue

        indices = digit_indices[:samples_per_class]

        for idx, img_idx in enumerate(indices):
            ax = plt.subplot(10, samples_per_class, digit * samples_per_class + idx + 1)
            img = images[img_idx].squeeze()
            plt.imshow(img.cpu(), cmap='gray')
            plt.axis('off')
            if idx == 0:
                plt.title(f'Digit: {digit}')

    plt.tight_layout()
    plt.show()

class DigitRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition Dataset Creator")

        # Initialize drawing variables
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.selected_digit = None

        # Initialize digit counters
        self.digit_counts = {i: 0 for i in range(10)}
        self.DIGITS_REQUIRED = 10  # Number of samples needed per digit

        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create digit selection buttons and labels
        digit_frame = ttk.Frame(main_frame)
        digit_frame.grid(row=0, column=0, columnspan=2, pady=5)

        self.digit_buttons = []
        self.count_labels = []
        for i in range(10):
            btn = ttk.Button(digit_frame, text=str(i), width=3,
                           command=lambda x=i: self.select_digit(x))
            btn.grid(row=0, column=i, padx=2)
            self.digit_buttons.append(btn)

            label = ttk.Label(digit_frame, text=f"0/{self.DIGITS_REQUIRED}")
            label.grid(row=1, column=i, padx=2)
            self.count_labels.append(label)

        # Create canvas for drawing
        self.canvas = tk.Canvas(main_frame, width=280, height=280, bg='white',
                              cursor='cross')
        self.canvas.grid(row=1, column=0, columnspan=2, pady=5)

        # Create preview canvas
        self.preview_canvas = tk.Canvas(main_frame, width=140, height=140, bg='white')
        self.preview_canvas.grid(row=1, column=2, pady=5, padx=5)
        preview_label = ttk.Label(main_frame, text="Preview")
        preview_label.grid(row=0, column=2)

        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)

        # Create control buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=2, column=0, columnspan=3, pady=5)

        ttk.Button(btn_frame, text="Clear Canvas",
                  command=self.clear_canvas).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Preview",
                  command=self.update_preview).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Save",
                  command=self.save_drawing).grid(row=0, column=2, padx=5)

        # Create PIL image for drawing
        self.image = Image.new('RGB', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # Storage for collected images
        self.collected_images = []
        self.collected_labels = []

    def center_and_scale_digit(self, image):
        """Center the digit and scale it appropriately"""
        # Convert to grayscale if not already
        img_gray = image if image.mode == 'L' else image.convert('L')

        # Find the bounding box of the digit
        # Convert black pixels to white and vice versa for bbox detection
        inverted = Image.eval(img_gray, lambda x: 255 - x)
        bbox = inverted.getbbox()

        if not bbox:
            return img_gray  # Return original if no digit found

        # Add padding to bounding box
        padding = 20
        left, top, right, bottom = bbox
        width = right - left
        height = bottom - top

        # Ensure square aspect ratio
        size = max(width, height) + 2 * padding

        # Calculate new bounding box
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2

        new_left = center_x - size // 2
        new_top = center_y - size // 2
        new_right = new_left + size
        new_bottom = new_top + size

        # Ensure bounds are within image
        new_left = max(0, new_left)
        new_top = max(0, new_top)
        new_right = min(img_gray.width, new_right)
        new_bottom = min(img_gray.height, new_bottom)

        # Crop to padded square
        digit = img_gray.crop((new_left, new_top, new_right, new_bottom))

        return digit

    def process_image(self, image, preview=False):
        """Process the image to match MNIST style"""
        # Convert to grayscale
        img_gray = image.convert('L')

        # Center and scale the digit
        img_centered = self.center_and_scale_digit(img_gray)

        # Calculate target size
        target_size = 140 if preview else 28

        # Resize image
        img_resized = img_centered.resize((target_size, target_size),
                                        Image.Resampling.LANCZOS)

        # Apply Gaussian blur to smooth edges
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=0.5))

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img_blurred)
        img_enhanced = enhancer.enhance(2.0)

        # Normalize pixel values
        def normalize_pixels(x):
            threshold = 200
            return 255 if x > threshold else max(0, x - 50)

        img_normalized = Image.eval(img_enhanced, normalize_pixels)

        # Invert colors for MNIST style (black background, white digit)
        img_final = Image.eval(img_normalized, lambda x: 255 - x)

        return img_final

    def update_preview(self):
        """Update the preview canvas with processed digit"""
        # Process the image for preview
        preview_img = self.process_image(self.image, preview=True)

        # Convert to PhotoImage for canvas
        self.preview_photo = ImageTk.PhotoImage(preview_img)

        # Clear previous preview and display new one
        self.preview_canvas.delete('all')
        self.preview_canvas.create_image(70, 70, image=self.preview_photo)

    def select_digit(self, digit):
        self.selected_digit = digit
        for btn in self.digit_buttons:
            btn.state(['!pressed'])
        self.digit_buttons[digit].state(['pressed'])

    def start_drawing(self, event):
        if self.selected_digit is None:
            messagebox.showwarning("Warning", "Please select a digit first!")
            return
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if self.drawing:
            x = event.x
            y = event.y
            if self.last_x and self.last_y:
                self.canvas.create_line((self.last_x, self.last_y, x, y),
                                     width=20, fill='black',
                                     capstyle=tk.ROUND, smooth=True)
                self.draw.line([self.last_x, self.last_y, x, y],
                             fill='black', width=20)
            self.last_x = x
            self.last_y = y

    def stop_drawing(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None
        # Update preview after drawing is complete
        self.update_preview()

    def clear_canvas(self):
        self.canvas.delete('all')
        self.preview_canvas.delete('all')
        self.image = Image.new('RGB', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def save_drawing(self):
        if self.selected_digit is None:
            messagebox.showwarning("Warning", "Please select a digit first!")
            return

        if self.digit_counts[self.selected_digit] >= self.DIGITS_REQUIRED:
            messagebox.showinfo("Info", f"Already have enough samples for digit {self.selected_digit}")
            return

        # Process the image
        processed_img = self.process_image(self.image, preview=False)

        # Convert to tensor
        img_tensor = torch.tensor(list(processed_img.getdata()), dtype=torch.float32)
        img_tensor = img_tensor.reshape(1, 28, 28) / 255.0  # Normalize to [0, 1]

        # Store the image and label
        self.collected_images.append(img_tensor)
        self.collected_labels.append(self.selected_digit)

        # Update count and label
        self.digit_counts[self.selected_digit] += 1
        self.count_labels[self.selected_digit].config(
            text=f"{self.digit_counts[self.selected_digit]}/{self.DIGITS_REQUIRED}")

        # Clear canvas after saving
        self.clear_canvas()

        # Check if we have all required digits
        if all(count >= self.DIGITS_REQUIRED for count in self.digit_counts.values()):
            self.create_dataset()

    def create_dataset(self):
        messagebox.showinfo("Info", "Creating augmented dataset...")

        # Convert lists to tensors
        images = torch.stack(self.collected_images)
        labels = torch.tensor(self.collected_labels, dtype=torch.long)

        # Create augmentations
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Calculate number of augmentations needed
        num_augmentations = 99  # To get 1000 total (10 original + 990 augmented)

        augmented_images = []
        augmented_labels = []

        # Keep original images
        augmented_images.extend(self.collected_images)
        augmented_labels.extend(self.collected_labels)

        # Generate augmentations
        dataset = EMNISTAugmenter(images, labels)

        for _ in tqdm(range(num_augmentations), desc="Generating augmentations"):
            for img, label in dataset:
                augmented_images.append(img)
                augmented_labels.append(label)

        # Convert to tensors
        final_images = torch.stack(augmented_images)
        final_labels = torch.tensor(augmented_labels, dtype=torch.long)

        # Save the dataset
        save_dir = Path('datasets/noveldigits')
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(final_images, save_dir / 'novel_digits_images.pt')
        torch.save(final_labels, save_dir / 'novel_digits_labels.pt')

        messagebox.showinfo("Success",
                            "Dataset created successfully!\n"
                            f"Total images: {len(final_images)}\n"
                            "Files saved in 'datasets/noveldigits' directory")

        # Display some examples
        display_examples(final_images, final_labels, "Novel Digits Dataset Examples")

        # Close the application
        self.root.destroy()

def create_or_load_novel_digits_dataset():
    """Function to create or load the novel digits dataset"""
    save_dir = Path('datasets/noveldigits')
    images_path = save_dir / 'novel_digits_images.pt'
    labels_path = save_dir / 'novel_digits_labels.pt'

    # Check if dataset already exists
    if images_path.exists() and labels_path.exists():
        print("Dataset already exists, loading...")
        images = torch.load(images_path)
        labels = torch.load(labels_path)

        print(f"Dataset loaded successfully!")
        print(f"Images tensor shape: {images.shape}")
        print(f"Labels tensor shape: {labels.shape}")

        # Display examples
        display_examples(images, labels, "Existing Novel Digits Dataset Examples")

        return images, labels

    else:
        print("Dataset not found, starting creation process...")
        root = tk.Tk()
        app = DigitRecognitionApp(root)
        root.mainloop()
