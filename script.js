const navToggle = document.querySelector(".nav-toggle");
const navLinks = document.querySelector("#nav-links");
const lightbox = document.querySelector("#image-lightbox");
const lightboxImage = document.querySelector("#lightbox-image");
const toast = document.querySelector("#copy-toast");

navToggle?.addEventListener("click", () => {
  const open = !document.body.classList.contains("nav-open");
  document.body.classList.toggle("nav-open", open);
  navToggle.setAttribute("aria-expanded", String(open));
});

navLinks?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    document.body.classList.remove("nav-open");
    navToggle?.setAttribute("aria-expanded", "false");
  });
});

const revealItems = document.querySelectorAll(".reveal");
const scoreBars = document.querySelectorAll(".chart-track i[data-score]");

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function showBar(bar) {
  bar.style.width = `${bar.dataset.score}%`;
}

function revealAll() {
  revealItems.forEach((item) => item.classList.add("is-visible"));
  scoreBars.forEach(showBar);
}

if (reducedMotion.matches || !("IntersectionObserver" in window)) {
  revealAll();
} else {
  const seenPerGroup = new Map();
  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const item = entry.target;
        const group = item.parentElement;
        const step = seenPerGroup.get(group) || 0;
        seenPerGroup.set(group, step + 1);
        item.style.setProperty("--reveal-delay", `${Math.min(step, 4) * 70}ms`);
        item.classList.add("is-visible");
        item.querySelectorAll(".chart-track i[data-score]").forEach(showBar);
        revealObserver.unobserve(item);
      });
    },
    { rootMargin: "0px 0px -10% 0px", threshold: 0.1 }
  );

  revealItems.forEach((item) => revealObserver.observe(item));
  scoreBars.forEach((bar) => {
    if (!bar.closest(".reveal")) showBar(bar);
  });

  // Safety net: never leave content hidden if the observer never fires.
  window.setTimeout(() => {
    if (document.querySelector(".reveal:not(.is-visible)")) revealAll();
  }, 3000);
}

document.querySelectorAll(".zoom-trigger").forEach((trigger) => {
  trigger.addEventListener("click", () => {
    if (!lightbox || !lightboxImage) return;
    lightboxImage.src = trigger.dataset.image || "";
    lightboxImage.alt = trigger.querySelector("img")?.alt || "";
    document.body.classList.add("dialog-open");
    if (typeof lightbox.showModal === "function") {
      lightbox.showModal();
    } else {
      lightbox.setAttribute("open", "");
    }
  });
});

const taskSwitcher = document.querySelector(".task-switcher");
const taskButtons = document.querySelectorAll(".task-tab");
const taskSheets = document.querySelectorAll(".task-sheet");
const taskViewport = document.querySelector(".task-viewport");
const taskCopy = document.querySelector("[data-task-copy]");

const taskDetails = {
  packing: {
    image: "assets/svg/qualitative-packing.svg",
    copy: "BAGEL adds irrelevant accessories, OmniGen2 repeats checklist images, and Janus-Pro produces illegible in-image text. UniWeaver keeps clothing and equipment advice tied to a distinct image for each unit.",
  },
  cooking: {
    image: "assets/svg/qualitative-cooking.svg",
    copy: "BAGEL misses key steps, OmniGen2 produces an unreadable shopping list, and ILLUME+ repeats similar scenes and instructions. UniWeaver moves from preparation to baking and serving, with temperature, timing, and a separate image for each stage.",
  },
  architecture: {
    image: "assets/svg/final_appendix_fig.svg",
    copy: "BAGEL repeats similar exterior views, TokenFlow repeats planning-oriented text, and ILLUME+ exposes intermediate instructions. UniWeaver gives the entrance, rehearsal room, and rooftop garden their own units, then explains how they connect.",
  },
};

taskButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const task = button.dataset.task;
    if (!taskDetails[task] || !taskSwitcher) return;
    taskSwitcher.dataset.currentTask = task;
    taskButtons.forEach((item) => {
      const active = item === button;
      item.classList.toggle("is-active", active);
      item.setAttribute("aria-pressed", String(active));
    });
    taskSheets.forEach((sheet) => {
      sheet.classList.toggle("is-active", sheet.getAttribute("src") === taskDetails[task].image);
    });
    if (taskViewport) {
      taskViewport.dataset.image = taskDetails[task].image;
    }
    if (taskCopy) taskCopy.textContent = taskDetails[task].copy;
  });
});

function closeLightbox() {
  if (!lightbox?.open) return;
  lightbox.close();
}

document.querySelector("[data-close-lightbox]")?.addEventListener("click", closeLightbox);
lightbox?.addEventListener("click", (event) => {
  if (event.target === lightbox) closeLightbox();
});
lightbox?.addEventListener("close", () => {
  document.body.classList.remove("dialog-open");
  lightboxImage?.removeAttribute("src");
});

function showToast(message = "BibTeX copied") {
  if (!toast) return;
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.clearTimeout(showToast.timeout);
  showToast.timeout = window.setTimeout(() => toast.classList.remove("is-visible"), 1800);
}

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}

document.querySelectorAll("[data-copy-target]").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copyTarget);
    if (!target) return;
    try {
      await copyText(target.textContent.trim());
      showToast();
    } catch {
      showToast("Copy failed. Select the BibTeX manually.");
    }
  });
});

const sections = [...document.querySelectorAll("main section[id]")];
const navAnchors = [...document.querySelectorAll(".nav-links a[href^='#']")];
const siteHeader = document.querySelector(".site-header");
const readingProgress = document.querySelector("[data-reading-progress]");
const backToTop = document.querySelector("[data-back-to-top]");
let navFrame = 0;

backToTop?.addEventListener("click", () => {
  window.scrollTo({ top: 0, behavior: "smooth" });
});

function updateNavSection() {
  navFrame = 0;
  const scrollableHeight = document.documentElement.scrollHeight - window.innerHeight;
  const progress = scrollableHeight > 0 ? Math.min(1, Math.max(0, window.scrollY / scrollableHeight)) : 0;
  if (readingProgress) readingProgress.style.transform = `scaleX(${progress})`;
  backToTop?.classList.toggle("is-visible", window.scrollY > window.innerHeight * 0.75);

  const atPageEnd =
    Math.ceil(window.scrollY + window.innerHeight) >=
    document.documentElement.scrollHeight - 1;
  const threshold = (siteHeader?.getBoundingClientRect().height || 76) + 1;
  let currentId = atPageEnd ? "cite" : null;

  if (!atPageEnd) {
    sections.forEach((section) => {
      if (section.getBoundingClientRect().top <= threshold) currentId = section.id;
    });
  }

  navAnchors.forEach((link) => {
    if (link.getAttribute("href") === `#${currentId}`) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });
}

function scheduleNavUpdate() {
  if (!navFrame) navFrame = window.requestAnimationFrame(updateNavSection);
}

window.addEventListener("scroll", scheduleNavUpdate, { passive: true });
window.addEventListener("resize", scheduleNavUpdate);
updateNavSection();

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && document.body.classList.contains("nav-open") && !lightbox?.open) {
    document.body.classList.remove("nav-open");
    navToggle?.setAttribute("aria-expanded", "false");
  }
});
