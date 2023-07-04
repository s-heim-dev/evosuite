/*
 * Copyright (C) 2010-2018 Gordon Fraser, Andrea Arcuri and EvoSuite
 * contributors
 *
 * This file is part of EvoSuite.
 *
 * EvoSuite is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3.0 of the License, or
 * (at your option) any later version.
 *
 * EvoSuite is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with EvoSuite. If not, see <http://www.gnu.org/licenses/>.
 */
package org.evosuite.ga.metaheuristics;

import org.evosuite.Properties;
import org.evosuite.TimeController;
import org.evosuite.ga.*;
import org.evosuite.utils.Randomness;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Implementation of steady state GA
 *
 * @author Gordon Fraser
 */
public class MonotonicGA<T extends Chromosome<T>> extends GeneticAlgorithm<T> {

    private static final long serialVersionUID = 7846967347821123201L;

    protected ReplacementFunction<T> replacementFunction;

    private final Logger logger = LoggerFactory.getLogger(MonotonicGA.class);

    /**
     * Constructor
     *
     * @param factory a {@link org.evosuite.ga.ChromosomeFactory} object.
     */
    public MonotonicGA(ChromosomeFactory<T> factory) {
        super(factory);

        setReplacementFunction(new FitnessReplacementFunction<>());
    }

    /**
     * <p>
     * keepOffspring
     * </p>
     *
     * @param parents    a list of {@link org.evosuite.ga.Chromosome} objects.
     * @param offsprings    a list of {@link org.evosuite.ga.Chromosome} objects.
     * @return a boolean.
     */
    protected boolean keepOffspring(List<T> parents, List<T> offsprings) {
        return replacementFunction.keepOffspring(parents, offsprings);
    }

    /**
     * {@inheritDoc}
     */
    @SuppressWarnings("unchecked")
    @Override
    protected void evolve(int parentsNumber) {

        // Elitism
        logger.debug("Elitism");
        List<T> newGeneration = new ArrayList<>(elitism());

        // Add random elements
        // new_generation.addAll(randomism());

        while (!isNextPopulationFull(newGeneration) && !isFinished()) {
            logger.debug("Generating offspring");

            List<T> parents = new ArrayList<>();
            List<T> offsprings = new ArrayList<>();

            T parent1 = selectionFunction.select(population);
            parents.add(parent1);
            offsprings.add(parent1.clone());

            for (int i = 1; i < parentsNumber; i++) {
                T parent2;
                if (Properties.HEADLESS_CHICKEN_TEST)
                    parent2 = newRandomIndividual(); // crossover with new random individual
                else
                    parent2 = selectionFunction.select(population); // crossover with existing individual
                
                parents.add(parent2);
                offsprings.add(parent2.clone());
            }

            try {
                // Crossover
                if (Randomness.nextDouble() <= Properties.CROSSOVER_RATE) {
                    crossoverFunction.crossOver(offsprings);
                }

            } catch (ConstructionFailedException e) {
                logger.info("CrossOver failed");
                continue;
            }

            for (T offspring : offsprings) {
                // Mutation
                notifyMutation(offspring);
                offspring.mutate();

                if (offspring.isChanged()) {
                    offspring.updateAge(currentIteration);
                }

                // The offspring replace the parents if and only if one of
                // the offspring is not worse than the best parent.
                for (FitnessFunction<T> fitnessFunction : fitnessFunctions) {
                    fitnessFunction.getFitness(offspring);
                    notifyEvaluation(offspring);
                }
            }

            if (keepOffspring(parents, offsprings)) {
                logger.debug("Keeping offspring");

                // Reject offspring straight away if it's too long
                int rejected = 0;

                for (T offspring : offsprings) {
                    if (isTooLong(offspring) || offspring.size() == 0) {
                        rejected++;
                    } else {
                        // if(Properties.ADAPTIVE_LOCAL_SEARCH ==
                        // AdaptiveLocalSearchTarget.ALL)
                        // applyAdaptiveLocalSearch(offspring);
                        newGeneration.add(offspring);
                    }
                }

                for (int i = 0; i < rejected; i++) {
                    T parent = Randomness.choice(parents);
                    parents.remove(parent);
                    newGeneration.add(parent);
                }
            } else {
                logger.debug("Keeping parents");

                for (T parent : parents) {
                    newGeneration.add(parent);
                }
            }

        }

        population = newGeneration;
        // archive
        updateFitnessFunctionsAndValues();

        currentIteration++;
    }

    private T newRandomIndividual() {
        T randomChromosome = chromosomeFactory.getChromosome();
        for (FitnessFunction<T> fitnessFunction : this.fitnessFunctions) {
            randomChromosome.addFitness(fitnessFunction);
        }
        return randomChromosome;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initializePopulation() {
        notifySearchStarted();
        currentIteration = 0;

        // Set up initial population
        generateInitialPopulation(Properties.POPULATION);
        logger.debug("Calculating fitness of initial population");
        calculateFitnessAndSortPopulation();

        this.notifyIteration();
    }

    private static final double DELTA = 0.000000001; // it seems there is some
    // rounding error in LS,
    // but hard to debug :(

    /**
     * {@inheritDoc}
     */
    @Override
    public void generateSolution() {
        if (Properties.ENABLE_SECONDARY_OBJECTIVE_AFTER > 0 || Properties.ENABLE_SECONDARY_OBJECTIVE_STARVATION) {
            disableFirstSecondaryCriterion();
        }

        if (population.isEmpty()) {
            initializePopulation();
            assert !population.isEmpty() : "Could not create any test";
        }

        logger.debug("Starting evolution");
        int starvationCounter = 0;
        double bestFitness = Double.MAX_VALUE;
        double lastBestFitness = Double.MAX_VALUE;
        if (getFitnessFunction().isMaximizationFunction()) {
            bestFitness = 0.0;
            lastBestFitness = 0.0;
        }

        while (!isFinished()) {

            logger.info("Population size before: " + population.size());
            // related to Properties.ENABLE_SECONDARY_OBJECTIVE_AFTER;
            // check the budget progress and activate a secondary criterion
            // according to the property value.

            {
                double bestFitnessBeforeEvolution = getBestFitness();
                evolve();
                sortPopulation();
                double bestFitnessAfterEvolution = getBestFitness();

                if (getFitnessFunction().isMaximizationFunction())
                    assert (bestFitnessAfterEvolution >= (bestFitnessBeforeEvolution
                            - DELTA)) : "best fitness before evolve()/sortPopulation() was: " + bestFitnessBeforeEvolution
                            + ", now best fitness is " + bestFitnessAfterEvolution;
                else
                    assert (bestFitnessAfterEvolution <= (bestFitnessBeforeEvolution
                            + DELTA)) : "best fitness before evolve()/sortPopulation() was: " + bestFitnessBeforeEvolution
                            + ", now best fitness is " + bestFitnessAfterEvolution;
            }

            {
                double bestFitnessBeforeLocalSearch = getBestFitness();
                applyLocalSearch();
                double bestFitnessAfterLocalSearch = getBestFitness();

                if (getFitnessFunction().isMaximizationFunction())
                    assert (bestFitnessAfterLocalSearch >= (bestFitnessBeforeLocalSearch
                            - DELTA)) : "best fitness before applyLocalSearch() was: " + bestFitnessBeforeLocalSearch
                            + ", now best fitness is " + bestFitnessAfterLocalSearch;
                else
                    assert (bestFitnessAfterLocalSearch <= (bestFitnessBeforeLocalSearch
                            + DELTA)) : "best fitness before applyLocalSearch() was: " + bestFitnessBeforeLocalSearch
                            + ", now best fitness is " + bestFitnessAfterLocalSearch;
            }

            /*
             * TODO: before explanation: due to static state handling, LS can
             * worse individuals. so, need to re-sort.
             *
             * now: the system tests that were failing have no static state...
             * so re-sorting does just hide the problem away, and reduce
             * performance (likely significantly). it is definitively a bug
             * somewhere...
             */
            // sortPopulation();

            double newFitness = getBestFitness();

            if (getFitnessFunction().isMaximizationFunction())
                assert (newFitness >= (bestFitness - DELTA)) : "best fitness was: " + bestFitness
                        + ", now best fitness is " + newFitness;
            else
                assert (newFitness <= (bestFitness + DELTA)) : "best fitness was: " + bestFitness
                        + ", now best fitness is " + newFitness;
            bestFitness = newFitness;

            if (Double.compare(bestFitness, lastBestFitness) == 0) {
                starvationCounter++;
            } else {
                logger.info("reset starvationCounter after " + starvationCounter + " iterations");
                starvationCounter = 0;
                lastBestFitness = bestFitness;

            }

            updateSecondaryCriterion(starvationCounter);

            logger.info("Current iteration: " + currentIteration);
            this.notifyIteration();

            logger.info("Population size: " + population.size());
            logger.info("Best individual has fitness: " + population.get(0).getFitness());
            logger.info("Worst individual has fitness: " + population.get(population.size() - 1).getFitness());

        }
        // archive
        TimeController.execute(this::updateBestIndividualFromArchive, "update from archive", 5_000);

        notifySearchFinished();
    }

    private double getBestFitness() {
        T bestIndividual = getBestIndividual();
        for (FitnessFunction<T> ff : fitnessFunctions) {
            ff.getFitness(bestIndividual);
        }
        return bestIndividual.getFitness();
    }

    /**
     * <p>
     * setReplacementFunction
     * </p>
     *
     * @param replacement_function a {@link org.evosuite.ga.ReplacementFunction} object.
     */
    public void setReplacementFunction(ReplacementFunction<T> replacement_function) {
        this.replacementFunction = replacement_function;
    }

    /**
     * <p>
     * getReplacementFunction
     * </p>
     *
     * @return a {@link org.evosuite.ga.ReplacementFunction} object.
     */
    public ReplacementFunction<T> getReplacementFunction() {
        return replacementFunction;
    }

}
