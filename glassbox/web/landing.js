(() => {
    const spooky = document.getElementById('landing-spooky');
    spooky?.addEventListener('change', () => {
        document.body.classList.toggle('spooky-mode', spooky.checked);
    });

    const copyButton = document.getElementById('copy-command');
    copyButton?.addEventListener('click', async () => {
        try {
            await navigator.clipboard?.writeText(document.getElementById('pip-command').textContent);
        } catch (_) {}
        copyButton.textContent = 'Copied';
        setTimeout(() => { copyButton.textContent = 'Copy'; }, 1600);
    });

    document.querySelectorAll('.mock-heatmap .heatmap-grid').forEach(grid => {
        for (let i = 0; i < 80; i++) grid.appendChild(document.createElement('i'));
    });

    if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        const observer = new IntersectionObserver(entries => {
            entries.forEach(entry => entry.target.classList.toggle('is-visible', entry.isIntersecting));
        }, { threshold: 0.45 });
        document.querySelectorAll('.scroll-card').forEach(card => observer.observe(card));
    } else {
        document.querySelectorAll('.scroll-card').forEach(card => card.classList.add('is-visible'));
    }
})();