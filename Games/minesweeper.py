"""
Minesweeper Game

The difficulty levels, the number of mines, the size of the field as per difficulty is taken from:
"https://minesweepergame.com/strategy/how-to-play-minesweeper.php"

Difficulties:
Beginner (8x8 or 9x9 with 10 mines)
Intermediate (16x16 with 40 mines)
Expert (30x16 with 99 mines)

Steps to generate Minefield:
1. Input difficulty level
2. Generate the Minefield based on the selected difficulty level (This will generate a 2x2 Matrix)
3. Add the mines randomly across the Minefield (Matrix)
4. The mines will be initialized as -1
5. Every "square" on the field have an integer as information to point out the number of mines 
that are surrounding that "square"

Steps for Player:
1. Input coordinate of a "square"
2. 

WORK IN PROGRESS
"""

