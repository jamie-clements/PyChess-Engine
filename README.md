# PyChess Engine

## Overview
PyChess Engine is a Python-based chess engine built with a focus on chess rule implementation, AI opponent logic, and a visually appealing graphical user interface using Pygame. This project caters to chess enthusiasts and developers looking to explore the intricacies of chess programming, including move generation, board evaluation, and game rules.

## Features
- **Complete Chess Rules**:
  - Piece-specific movements.
  - Special rules like castling, en passant, and pawn promotion.
  - Check and checkmate detection.
- **AI Opponent**:
  - Three difficulty levels (Easy, Medium, Hard) powered by the Minimax algorithm with alpha-beta pruning.
- **Graphical User Interface (GUI)**:
  - Interactive chessboard built with Pygame.
  - Drag-and-drop functionality for human moves.
- **Modular Design**:
  - Flexible architecture for easy improvements and feature additions.

## How to Play
1. Install the required Python dependencies:
   ```bash
   pip install pygame numpy
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pychess-engine.git
   ```
3. Navigate to the project directory:
   ```bash
   cd pychess-engine
   ```
4. Run the game:
   ```bash
   python chess_gui.py
   ```
5. Play against the AI or test the engine's move logic.

## Installation
Ensure you have Python 3.7 or higher installed. Then, install the necessary dependencies as outlined in the **How to Play** section.

## Future Enhancements
- Integration of an opening book for smarter AI play.
- Implementation of an endgame tablebase.
- Online multiplayer support.
- Improved evaluation function for better AI decision-making.

## Contributing
Contributions are welcome! Feel free to fork this repository, make your improvements, and submit a pull request. Ensure your contributions adhere to Python best practices and include appropriate comments and documentation.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
PyChess Engine is inspired by the beauty of chess and the challenge of simulating its rules and strategies programmatically. Special thanks to the Pygame and NumPy libraries for their role in building this project.

