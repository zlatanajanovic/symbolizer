const toggle = document.querySelector('.nav-toggle');
const links = document.querySelector('.nav-links');

toggle?.addEventListener('click', () => {
  const open = toggle.getAttribute('aria-expanded') === 'true';
  toggle.setAttribute('aria-expanded', String(!open));
  links.classList.toggle('is-open', !open);
});

links?.addEventListener('click', (event) => {
  if (event.target.matches('a')) {
    toggle?.setAttribute('aria-expanded', 'false');
    links.classList.remove('is-open');
  }
});

function configureRepositoryLink() {
  const container = document.querySelector('[data-repository-url]');
  const pending = container?.querySelector('.repository-status');
  const link = container?.querySelector('.repository-link');
  const value = container?.dataset.repositoryUrl?.trim();

  if (!container || !pending || !link || !value || (value.includes('{{') && value.includes('}}'))) return;

  try {
    const url = new URL(value);
    if (!['http:', 'https:'].includes(url.protocol)) return;
    link.href = url.href;
    link.hidden = false;
    pending.hidden = true;
  } catch (_) {
    // Keep the honest pending state for an incomplete or malformed substitution.
  }
}

window.configureRepositoryLink = configureRepositoryLink;
configureRepositoryLink();
