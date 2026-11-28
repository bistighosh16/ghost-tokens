(() => {
    const body = document.body;
    const spookyToggle = document.getElementById('spooky-mode');
    const copyButton = document.getElementById('copy-install');
    const copyStatus = document.getElementById('copy-status');
    const cards = [...document.querySelectorAll('[data-tour-step]')];
    const progressLinks = [...document.querySelectorAll('.tour-progress a')];

    spookyToggle.addEventListener('change', () => {
        body.classList.toggle('spooky-on', spookyToggle.checked);
        body.classList.toggle('spooky-off', !spookyToggle.checked);
    });

    copyButton.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(document.getElementById('install-command').textContent);
            copyStatus.textContent = 'Copied to clipboard.';
            copyButton.textContent = 'Copied';
        } catch (error) {
            copyStatus.textContent = 'Copy unavailable here. Select the command to copy it.';
            console.error('Could not copy the install command:', error);
        }
        window.setTimeout(() => { copyButton.textContent = 'Copy'; }, 1800);
    });

    if (!('IntersectionObserver' in window)) {
        cards.forEach(card => card.classList.add('is-active'));
        return;
    }

    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            const step = Number(entry.target.dataset.tourStep);
            entry.target.classList.toggle('is-active', entry.isIntersecting);
            const link = progressLinks[step - 1];
            if (link && entry.isIntersecting) {
                progressLinks.forEach(item => item.removeAttribute('aria-current'));
                link.setAttribute('aria-current', 'step');
            }
        });
    }, { threshold: 0.35 });
    cards.forEach(card => observer.observe(card));
})();
