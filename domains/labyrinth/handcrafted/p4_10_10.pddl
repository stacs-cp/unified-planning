(define (problem labyrinth-size-4-rotations-10-seed-10)
(:domain labyrinth)
(:objects
	pos0 pos1 pos2 pos3  - gridpos
	card0 card1 card2 card3 card4 card5 card6 card7 card8 card9 card10 card11 card12 card13 card14 card15  - card
)
(:init
	(MAX-POS pos3)
	(MIN-POS pos0)

	(NEXT pos1 pos0)
	(NEXT pos2 pos1)
	(NEXT pos3 pos2)

	(card-at card0 pos0 pos0)
	(card-at card5 pos1 pos0)
	(card-at card14 pos2 pos0)
	(card-at card3 pos3 pos0)
	(card-at card7 pos0 pos1)
	(card-at card4 pos1 pos1)
	(card-at card8 pos2 pos1)
	(card-at card11 pos3 pos1)
	(card-at card13 pos0 pos2)
	(card-at card6 pos1 pos2)
	(card-at card10 pos2 pos2)
	(card-at card2 pos3 pos2)
	(card-at card12 pos0 pos3)
	(card-at card1 pos1 pos3)
	(card-at card9 pos2 pos3)
	(card-at card15 pos3 pos3)

	(BLOCKED card0 N)
	(BLOCKED card0 S)

	(BLOCKED card5 S)

	(BLOCKED card14 E)
	(BLOCKED card14 S)

	(BLOCKED card3 N)

	(BLOCKED card7 N)
	(BLOCKED card7 S)

	(BLOCKED card4 E)

	(BLOCKED card8 N)
	(BLOCKED card8 S)

	(BLOCKED card11 N)

	(BLOCKED card13 N)
	(BLOCKED card13 S)

	(BLOCKED card6 E)
	(BLOCKED card6 W)

	(BLOCKED card10 W)

	(BLOCKED card2 E)

	(BLOCKED card12 N)
	(BLOCKED card12 S)

	(BLOCKED card1 N)
	(BLOCKED card1 S)

	(BLOCKED card9 N)
	(BLOCKED card9 W)

	(BLOCKED card15 E)
	(BLOCKED card15 W)


	(robot-at card0)

	(= (total-cost) 0)
	(= (move-robot-cost) 1)
	(= (move-card) 1)
)
(:goal
	(and
		(left)
	)
)
	(:metric minimize (total-cost))
)
