(define (problem blocksworld_real12)
  (:domain blocksworld_real)
  (:objects
    blue - block
    yellow - block
    black - block
    white - block
  )
  (:init
    (clear black)
    (holding white)
	(handfull)
    (on yellow blue)
    (on black yellow)
    (ontable blue)
  )
  (:goal (and
    (on white black)
    (on black yellow)
    (on yellow blue)
  ))
)
