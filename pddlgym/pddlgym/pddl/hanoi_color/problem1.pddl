(define (problem hanoi1)
  (:domain hanoi)
  (:objects peg1 peg2 peg3 red_disk green_disk blue_disk yellow_disk)
  (:init
   (smaller peg1 red_disk) (smaller peg1 green_disk) (smaller peg1 blue_disk) (smaller peg1 yellow_disk)
   (smaller peg2 red_disk) (smaller peg2 green_disk) (smaller peg2 blue_disk) (smaller peg2 yellow_disk)
   (smaller peg3 red_disk) (smaller peg3 green_disk) (smaller peg3 blue_disk) (smaller peg3 yellow_disk)
   (smaller green_disk red_disk) (smaller blue_disk red_disk) (smaller blue_disk green_disk) (smaller yellow_disk red_disk)
   (smaller yellow_disk green_disk) (smaller yellow_disk blue_disk)
   (clear peg2) (clear peg3) (clear red_disk)
   (on yellow_disk peg1) (on blue_disk yellow_disk) (on green_disk blue_disk) (on red_disk green_disk)
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
  )
  (:goal (and (on yellow_disk peg3) (on blue_disk yellow_disk) (on green_disk blue_disk) (on red_disk green_disk)))
  )