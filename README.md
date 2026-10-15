# hsinyilin19.github.io

Personal research site of Hsin-Yi (Lucy) Lin: https://hsinyilin19.github.io. Plain HTML/CSS/JS, no build step.

[Demo page](https://hsinyilin19.github.io/Demo_adversarial_example_speech_quality/) for the paper: **On the robustness of non-intrusive speech quality model by adversarial examples (ICASSP 2023)**

## Publish on GitHub Pages
1. On GitHub, create a **public** repository named exactly `hsinyilin19.github.io`.
2. Upload everything in this folder (keep the `assets/` folder structure), or:
   ```bash
   git init && git add . && git commit -m "Personal site"
   git branch -M main
   git remote add origin https://github.com/hsinyilin19/hsinyilin19.github.io.git
   git push -u origin main
   ```
3. Repository → Settings → Pages → Source: "Deploy from a branch", branch `main`, folder `/ (root)`.
4. The site appears at https://hsinyilin19.github.io within a minute or two.

## Editing
- Text and publications: `index.html` (each paper is one `<li class="pub">`; `data-t` holds its topic tags: `q`, `ml`, `pde`, `mod`).
- Colors and type: `assets/style.css` (tokens at the top).
- Hero animation (flow on the torus): `assets/main.js`.
- Figures: `assets/fig/`. Files starting with `q_` are cropped from the quantum papers; `averaging`, `torus`, `migration`, `ot` are illustrations generated for this site.
- Math is rendered by KaTeX and the font is Newsreader, both self-hosted in `assets/vendor/` (MIT and OFL licenses).
