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
import org.evosuite.ga.Chromosome;
import org.evosuite.ga.ChromosomeFactory;
import org.evosuite.ga.ConstructionFailedException;
import org.evosuite.ga.FitnessFunction;
import org.evosuite.utils.Randomness;

import java.util.*;


/**
 * Alternative version of steady state GA
 *
 * @author Gordon Fraser
 */
public class SteadyStateGA<T extends Chromosome<T>> extends MonotonicGA<T> {

    private static final long serialVersionUID = 7301010503732698233L;

    private final org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger(SteadyStateGA.class);

    /**
     * Generate a new search object
     *
     * @param factory a {@link org.evosuite.ga.ChromosomeFactory} object.
     */
    public SteadyStateGA(ChromosomeFactory<T> factory) {
        super(factory);
    }

    /**
     * {@inheritDoc}
     * <p>
     * Perform one iteration of the search
     */
    @Override
    protected void evolve(int parentsNumber) {
        logger.debug("Generating offspring");
        currentIteration++;

        List<T> parents = new ArrayList<>();
        List<T> offsprings = new ArrayList<>();

        for (int i = 0; i < parentsNumber; i++) {
            T parent = selectionFunction.select(population);
            parents.add(parent);
            offsprings.add(parent.clone());
        }

        try {
            // Crossover
            if (Randomness.nextDouble() <= Properties.CROSSOVER_RATE) {
                crossoverFunction.crossOver(offsprings);
            }

            // Mutation
            for (T offspring : offsprings) {
                notifyMutation(offspring);
                offspring.mutate();

                if (offspring.isChanged()) {
                    offspring.updateAge(currentIteration);
                }
            }
        } catch (ConstructionFailedException e) {
            logger.info("CrossOver/Mutation failed");
            return;
        }

        // The two offspring replace the parents if and only if one of
        // the offspring is not worse than the best parent.
        for (FitnessFunction<T> fitnessFunction : fitnessFunctions) {
            for (T offspring : offsprings) {
                fitnessFunction.getFitness(offspring);
                notifyEvaluation(offspring);
            }
        }

        if (!Properties.PARENT_CHECK
                || keepOffspring(parents, offsprings)) {
            logger.debug("Keeping offspring");

            for (int i = 0; i < parentsNumber; i++) {
                T offspring = offsprings.get(i);
                if (!isTooLong(offspring)) {
                    population.remove(parents.get(i));
                    population.add(offspring);
                }
            }
        } else {
            logger.debug("Keeping parents");
        }

        updateFitnessFunctionsAndValues();
    }

}
