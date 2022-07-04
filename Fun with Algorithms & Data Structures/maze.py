"""
Recursive, Maze-like

A maze, a mouse and a cheese. 

The mouse starts at top-left corner, 
Purpose of this algorithm - to see if the mouse is able to reach the cheese and take a bite! 
Returns 1 (Success for the mouse) or 0 (Mouse unable to get to the cheese)

Simple example for this algorithm,
5 - mouse; 1 - Open Path/Route; 0 - Wall; 9 - Cheese;

5
1 0 0 1
1 0 9 1
1 1 1 0
1 0 0 0

Output: 1 - Yes it is able to bite the cheese!
"""
def escape(v, h, maze, truth):
    # Baseline
    if truth:
        return True
    else:
        if maze[v][h] == 9:
            return True
        elif maze[v][h] == 0:
            return False
        elif maze[v][h] == 1:
            if v + 1 != len(maze):
                # Move Bottom
                truth = escape(v + 1, h, maze, truth)
                if truth:
                    return True
            if h + 1 != len(maze[v]):
                # Move Right
                truth = escape(v, h + 1, maze, truth)
                if truth:
                    return True
            try:
                # Check Top Route if there is cheese
                while v != 0:
                    v -= 1
                    # Hit wall
                    if maze[v][h] == 0:
                        return False
                    if maze[v][h] == 9:
                        return True
                    # Continue if 1 (Route)
            except:
                return False


if __name__ == '__main__':
    # Mouse start at top-left corner, 1 is route, 0 is wall, 9 is cheese
    maze = [[1,0,0,1], 
            [1,0,9,0], 
            [1,1,1,0], 
            [1,0,1,0]]
    answer = escape(0, 0, maze, False)
    # Output 1 is Yes (Able to get to the cheese), Output 0 is No
    output = 0
    if answer:
        output = 1
    print(output)