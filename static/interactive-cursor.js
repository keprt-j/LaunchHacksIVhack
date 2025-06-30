// Interactive grid cursor effect for dark backgrounds
// Moves the .grid-overlay background to follow the mouse

document.addEventListener('DOMContentLoaded', function() {
    const grid = document.querySelector('.grid-overlay');
    if (!grid) return;
    document.addEventListener('mousemove', function(e) {
        // Get mouse position relative to viewport
        const x = e.clientX;
        const y = e.clientY;
        // Move the grid background position to follow the mouse, with some smoothing
        grid.style.backgroundPosition = `${x - 100}px ${y - 100}px`;
    });
    // Optional: reset on mouse leave
    document.addEventListener('mouseleave', function() {
        grid.style.backgroundPosition = '';
    });
});
